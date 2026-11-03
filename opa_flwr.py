"""
opa_flwr.py - One-Shot Private Aggregation (OPA) Flower Client & Strategy for Genomic LM.
Based on "OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications
to Federated Learning" (CRYPTO 2025).

Components:
  1. OPAClient:
       - Trains local SimpleDNALM on genomic sequences.
       - Samples secret seed s in Z_q^lambda.
       - Generates mask = LWR_PRG.expand(s) in Z_p^L.
       - Masks quantized local weights: ciphertext = (w_quant + mask) mod p.
       - Splits seed s into m shares via Shamir's Secret Sharing over Fq.
       - Returns ciphertext as model parameters and shares in metrics["seed_shares"].

  2. OPAFedAvg (OPAStrategy):
       - Flower Server strategy overriding aggregate_fit().
       - Homomorphically aggregates SSS shares across clients: S_j = sum_i s_i^(j) mod q.
       - Reconstructs global seed sum_i s_i mod q via Lagrange interpolation.
       - Expands global mask = LWR_PRG.expand(sum_i s_i) mod p.
       - Sums all client ciphertexts, subtracts global mask mod p, and dequantizes to
         recover exact plaintext averaged global model weights.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

try:
    from OPA.crypto import (
        LWR_PRG,
        share,
        reconstruct,
        quantize_to_zp,
        dequantize_from_zp,
        DEFAULT_Q,
        DEFAULT_P,
        DEFAULT_SCALE,
    )
    from OPA.model import SimpleDNALM
except ImportError:
    try:
        from Genomic_privacy_audit.OPA.crypto import (
            LWR_PRG,
            share,
            reconstruct,
            quantize_to_zp,
            dequantize_from_zp,
            DEFAULT_Q,
            DEFAULT_P,
            DEFAULT_SCALE,
        )
        from Genomic_privacy_audit.OPA.model import SimpleDNALM
    except ImportError:
        from crypto import (
            LWR_PRG,
            share,
            reconstruct,
            quantize_to_zp,
            dequantize_from_zp,
            DEFAULT_Q,
            DEFAULT_P,
            DEFAULT_SCALE,
        )
        from model import SimpleDNALM


# =====================================================================
# 1. PURE OPA CLIENT
# =====================================================================

class OPAClient(NumPyClient):
    """
    Flower NumPyClient that unconditionally masks model updates via OPA.
    Zero plaintext weights are ever transmitted over the network.
    """

    def __init__(
        self,
        client_id: str,
        model: SimpleDNALM,
        loader: DataLoader,
        prg: LWR_PRG,
        local_epochs: int = 1,
        lr: float = 0.0005,
        scale: float = DEFAULT_SCALE,
        t: int = 3,
        r: int = 6,
        m: int = 10,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.prg = prg
        self.local_epochs = local_epochs
        self.lr = lr
        self.scale = scale
        self.t = t
        self.r = r
        self.m = m
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Update local model with current global parameters
        self.model.set_weights(parameters)

        # 2. Local Training via Cross-Entropy Loss
        self.model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        last_loss = 0.0
        for _ in range(self.local_epochs):
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), batch_y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                last_loss = loss.item()

        # 3. Extract trained local weights
        trained_weights = self.model.get_weights()
        flat_weights, shapes = self.prg.flatten_weights(trained_weights)

        # 4. OPA Cryptographic Masking
        # Sample random seed s in Z_q^lambda
        seed_s = np.random.randint(0, self.prg.q, size=self.prg.lam, dtype=np.int64)

        # Expand seed_s using LWR PRG to obtain pseudorandom mask in Z_p^L
        mask = self.prg.expand(seed_s)

        # Fixed-point quantization of weights to Z_p
        quantized_w = quantize_to_zp(flat_weights, scale=self.scale, p=self.prg.p)

        # Ciphertext: c = (w_quant + mask) mod p
        ciphertext_flat = (quantized_w + mask) % self.prg.p

        # Unflatten ciphertext back into Flower parameter list format
        ciphertext_weights = self.prg.unflatten_weights(ciphertext_flat.astype(np.float32), shapes)

        # 5. Shamir's Secret Sharing (SSS) of seed_s into m shares
        sss_shares = share(
            secret=seed_s,
            t=self.t,
            r=self.r,
            m=self.m,
            q=self.prg.q,
        )

        # Serialize shares to JSON string for metrics dictionary
        serialized_shares = json.dumps([[int(x), [int(val) for val in y]] for x, y in sss_shares])

        num_examples = len(self.loader.dataset) if hasattr(self.loader, "dataset") else len(self.loader)
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(last_loss),
            "seed_shares": serialized_shares,
        }

        return ciphertext_weights, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict]:
        self.model.set_weights(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits, _ = self.model(batch_x)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), batch_y.reshape(-1))
                total_loss += loss.item() * batch_x.size(0)
                total_tokens += batch_x.size(0)

        avg_loss = total_loss / max(total_tokens, 1)
        num_examples = len(self.loader.dataset) if hasattr(self.loader, "dataset") else len(self.loader)
        return float(avg_loss), num_examples, {"loss": float(avg_loss)}


# =====================================================================
# 2. PURE OPA STRATEGY (OPAFedAvg)
# =====================================================================

class OPAFedAvg(FedAvg):
    """
    Flower Strategy strictly executing OPA secure aggregation.
    Reconstructs the global seed from SSS shares and unmasks the aggregated ciphertexts.
    """

    def __init__(
        self,
        prg: LWR_PRG,
        scale: float = DEFAULT_SCALE,
        r: int = 6,
        m: int = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prg = prg
        self.scale = scale
        self.r = r
        self.m = m
        self.latest_global_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Optional[ClientProxy], FitRes]],
        failures: List[Union[Tuple[Optional[ClientProxy], FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        # 1. Extract masked ciphertexts and SSS shares
        client_ciphertexts_flat: List[np.ndarray] = []
        client_shares_list: List[List[Tuple[int, np.ndarray]]] = []
        shapes_reference = None

        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_c, shapes = self.prg.flatten_weights(client_ndarrays)
            if shapes_reference is None:
                shapes_reference = shapes
            client_ciphertexts_flat.append(flat_c.astype(np.int64) % self.prg.p)

            shares_raw = fit_res.metrics["seed_shares"]
            if isinstance(shares_raw, str):
                shares_decoded = json.loads(shares_raw)
            else:
                shares_decoded = shares_raw

            parsed_shares = [(int(x), np.array(y, dtype=np.int64)) for x, y in shares_decoded]
            client_shares_list.append(parsed_shares)

        num_clients = len(results)

        # 2. Sum SSS shares homomorphically across clients for r parties: S_j = sum_i s_i^(j) mod q
        combined_shares: List[Tuple[int, np.ndarray]] = []
        for party_idx in range(self.r):
            party_id = client_shares_list[0][party_idx][0]
            party_share_sum = np.zeros(self.prg.lam, dtype=np.int64)
            for c_shares in client_shares_list:
                party_share_sum = (party_share_sum + c_shares[party_idx][1]) % self.prg.q
            combined_shares.append((party_id, party_share_sum))

        # 3. Reconstruct global seed: s_global = sum_{i=1}^n s_i mod q
        global_seed = reconstruct(combined_shares, q=self.prg.q)

        # 4. Expand global seed to compute global mask: sum_i mask_i mod p
        global_mask = self.prg.expand(global_seed)

        # 5. Sum all client ciphertexts in Z_p: sum_i (w_i + mask_i) mod p
        total_ciphertext = np.zeros(self.prg.L, dtype=np.int64)
        for c in client_ciphertexts_flat:
            total_ciphertext = (total_ciphertext + c) % self.prg.p

        # 6. Unmask: w_sum_quant = (total_ciphertext - global_mask) mod p
        w_sum_quant = (total_ciphertext - global_mask) % self.prg.p

        # 7. Dequantize from Z_p and average across clients
        recovered_average_flat = dequantize_from_zp(
            weights_zp=w_sum_quant,
            scale=self.scale,
            p=self.prg.p,
            num_contributors=num_clients,
        )

        # 8. Unflatten back to model parameter list
        averaged_ndarrays = self.prg.unflatten_weights(recovered_average_flat, shapes_reference)
        self.latest_global_parameters = ndarrays_to_parameters(averaged_ndarrays)

        return self.latest_global_parameters, {"num_aggregated": num_clients}


# Strategy alias
OPAStrategy = OPAFedAvg

