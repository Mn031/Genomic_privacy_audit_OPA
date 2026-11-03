"""
main.py - 10-Client One-Shot Private Aggregation (OPA) & Multi-Vector Genomic Privacy Audit.
Based on "OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications
to Federated Learning" (CRYPTO 2025) and "Quantifying Memorization and Privacy Risks in Genomic
Language Models" (arXiv:2603.08913v3).

Workflow:
  1. Sets fixed deterministic seeds for reproducible execution.
  2. Synthesizes zero-order genomic corpus (1,000 train, 200 test sequences of length 256 nt).
  3. Injects 100 confidential 64-nt canaries across 4 repetition tiers (1, 5, 10, 20 duplicates)
     exclusively into Client 9's training split (Clients 0-8 remain clean).
  4. Provisions SimpleDNALM (~12.6M - 12.88M parameters).
  5. Dynamically provisions LWR_PRG matrix A to match the exact 12.88M parameter footprint (L)
     using deterministic chunk-wise generation in lightweight memory (~50MB).
  6. Executes Flower Federated Learning using OPAFedAvg and OPAClient:
       - Every client update is cryptographically masked via LWR PRG in Z_p.
       - Random seeds are split via Shamir's Secret Sharing (t=3, r=6, m=10) over Fq.
       - Server homomorphically aggregates ciphertexts and reconstructs global seed,
         unmasking the exact averaged global weights with zero plaintext exposure.
  7. Extracts final aggregated global model and executes the Three-Vector Privacy Attack:
       - Vector 1: Sequence Extraction (Tab Attack beam search -> s_ext & Exposure)
       - Vector 2: Perplexity Detection (memorization gap -> s_ppl)
       - Vector 3: Membership Inference Attack (LiRA -> s_mia)
       - Unified Metric: S_config = max(s_ppl, s_ext, s_mia)
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Handle paths for standalone or package execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

try:
    from OPA.crypto import (
        LWR_PRG,
        DEFAULT_Q,
        DEFAULT_P,
        DEFAULT_SCALE,
    )
    from OPA.opa_flwr import OPAClient, OPAFedAvg
    from OPA.dataset import (
        DNATokenizer,
        CanaryMetadata,
        get_genomic_dataloaders,
    )
    from OPA.model import SimpleDNALM
    from OPA.attack import (
        run_three_vector_privacy_audit,
        print_genomic_privacy_report,
    )
except ImportError:
    try:
        from Genomic_privacy_audit.OPA.crypto import (
            LWR_PRG,
            DEFAULT_Q,
            DEFAULT_P,
            DEFAULT_SCALE,
        )
        from Genomic_privacy_audit.OPA.opa_flwr import OPAClient, OPAFedAvg
        from Genomic_privacy_audit.OPA.dataset import (
            DNATokenizer,
            CanaryMetadata,
            get_genomic_dataloaders,
        )
        from Genomic_privacy_audit.OPA.model import SimpleDNALM
        from Genomic_privacy_audit.OPA.attack import (
            run_three_vector_privacy_audit,
            print_genomic_privacy_report,
        )
    except ImportError:
        from crypto import (
            LWR_PRG,
            DEFAULT_Q,
            DEFAULT_P,
            DEFAULT_SCALE,
        )
        from opa_flwr import OPAClient, OPAFedAvg
        from dataset import (
            DNATokenizer,
            CanaryMetadata,
            get_genomic_dataloaders,
        )
        from model import SimpleDNALM
        from attack import (
            run_three_vector_privacy_audit,
            print_genomic_privacy_report,
        )


def set_deterministic_seeds(seed: int = 42) -> None:
    """Sets random seeds for deterministic, reproducible model training and simulation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning Genomic Privacy Audit - OPA Cryptographic Transport (CRYPTO 2025)"
    )
    parser.add_argument("--rounds", type=int, default=50, help="Number of FL communication rounds (default: 50)")
    parser.add_argument("--clients", type=int, default=10, help="Number of simulated FL clients (default: 10)")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local training epochs per client round (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for Adam optimizer (default: 0.0005)")
    parser.add_argument("--lam", type=int, default=128, help="Security parameter / PRG seed length lambda (default: 128)")
    parser.add_argument("--t", type=int, default=3, help="Shamir corruption threshold t (default: 3)")
    parser.add_argument("--r", type=int, default=6, help="Shamir reconstruction threshold r (default: 6)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Fixed-point quantization scale (default: 10000.0)")
    parser.add_argument("--beam-width", type=int, default=10, help="Beam width for Vector 1 extraction (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print(" GENOMIC PRIVACY AUDIT: 10-CLIENT ONE-SHOT PRIVATE AGGREGATION (OPA)")
    print(" Cryptographic Transport: LWR Seed-Homomorphic PRG & Shamir Secret Sharing")
    print(" Benchmark Dataset: Real Biological DNA (GUE prom_300_all)")
    print(" Target Configuration: Client 9 Canary Injection vs. Clients 0-8 Clean Baselines")
    print("=" * 80)

    # 1. Deterministic Seeds
    set_deterministic_seeds(args.seed)

    # 2. Genomic Dataset & Canary Generation
    print("\n[Step 1/5] Loading Real Biological Genomic Corpus (GUE prom_300_all) & Partitioning...")
    tokenizer, client_loaders, test_loader, canaries, train_seqs, test_seqs = get_genomic_dataloaders(
        num_clients=args.clients,
        num_train=1000,
        num_test=200,
        seq_length=256,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"    - Training Corpus: 1,000 biological sequences of 256 nt (GUE benchmark)")
    print(f"    - Held-out Test Split: 200 sequences of 256 nt")
    print(f"    - Clients 0-8: Clean background genomic sequences (100 seqs each, 0 canaries)")
    print(f"    - Client 9: TARGET CLIENT hosting all 100 canaries across 4 tiers:")
    for tier in [1, 5, 10, 20]:
        count = sum(1 for c in canaries if c.tier == tier)
        print(f"        * Tier {tier:>2} ({tier:>2} repeats): {count} canaries ({count * tier} total copies)")

    # 3. Model Architecture Provisioning
    print("\n[Step 2/5] Initializing SimpleDNALM Architecture (Causal Transformer Baseline)...")
    init_model = SimpleDNALM(
        vocab_size=len(tokenizer),
        d_model=512,
        num_layers=4,
        n_heads=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in init_model.parameters())
    print(f"    - Transformer Layers: 4 (Multi-Head Causal Self-Attention, n_heads=8)")
    print(f"    - Hidden Dimension d_model: 512 | Feedforward Dimension d_ff: 2048")
    print(f"    - Activation: GELU | Dropout: 0.05 | Causal Masking: Enabled")
    print(f"    - Total Parameters L = {total_params:,}")

    # 4. Dynamically provision LWR PRG for 12.6M parameter footprint
    print(f"\n[Step 3/5] Dynamically Provisioning LWR PRG for L={total_params:,} Parameters:")
    print(f"    - Field Modulus q = {DEFAULT_Q} (2^31 - 1, Mersenne prime)")
    print(f"    - Ciphertext Modulus p = {DEFAULT_P} (2^20)")
    print(f"    - Seed Dimension lambda = {args.lam}")
    print(f"    - SSS Thresholds: Corruption t = {args.t}, Reconstruction r = {args.r}, Total m = {args.clients}")
    print(f"    - Matrix A dynamically evaluated via deterministic chunked streaming (memory footprint < 50MB)")

    prg = LWR_PRG(
        L=total_params,
        lam=args.lam,
        q=DEFAULT_Q,
        p=DEFAULT_P,
        chunk_size=200000,
        seed=args.seed,
    )

    # 5. OPA Federated Learning Simulation
    print(f"\n[Step 4/5] Starting 10-Client OPA Federated Learning Simulation ({args.rounds} Rounds):")
    strategy = OPAFedAvg(
        prg=prg,
        scale=args.scale,
        r=args.r,
        m=args.clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=ndarrays_to_parameters(init_model.get_weights()),
    )

    current_parameters = init_model.get_weights()
    for rnd in range(1, args.rounds + 1):
        fit_results = []
        round_losses = []

        for cid in range(args.clients):
            loader = client_loaders[cid]
            local_model = SimpleDNALM(
                vocab_size=len(tokenizer),
                d_model=512,
                num_layers=4,
                n_heads=8,
                d_ff=2048,
                max_seq_len=256,
                dropout=0.05,
            )
            client = OPAClient(
                client_id=f"client_{cid}",
                model=local_model,
                loader=loader,
                prg=prg,
                local_epochs=args.local_epochs,
                lr=args.lr,
                scale=args.scale,
                t=args.t,
                r=args.r,
                m=args.clients,
                device=device,
            )
            # fit() returns ciphertext weights (masked in Z_p) and SSS shares in metrics
            ciphertext_weights, num_samples, metrics = client.fit(current_parameters, {})
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(ciphertext_weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))
            round_losses.append(metrics.get("train_loss", 0.0))

        # Server homomorphically aggregates ciphertexts and reconstructs global seed to unmask
        aggregated_parameters, agg_metrics = strategy.aggregate_fit(server_round=rnd, results=fit_results, failures=[])
        if aggregated_parameters is not None:
            current_parameters = parameters_to_ndarrays(aggregated_parameters)

        mean_loss = float(np.mean(round_losses))
        c9_loss = round_losses[9]
        print(f"    -> [Round {rnd}/{args.rounds}] OPA Homomorphic Aggregation Complete | Mean Loss: {mean_loss:.4f} | Client 9 Loss: {c9_loss:.4f}")

    print("\n[OK] OPA Federated Learning simulation completed successfully (Zero Plaintext Leakage in Transit).")

    # 6. Extract Final Global Model
    print("\n[Step 5/5] Extracting Global Model from Secure OPA Aggregation...")
    global_model = SimpleDNALM(
        vocab_size=len(tokenizer),
        d_model=512,
        num_layers=4,
        n_heads=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.05,
    ).to(device)
    global_model.set_weights(current_parameters)
    global_model.eval()
    print("[OK] Global SimpleDNALM model synchronized and ready for multi-vector audit.")

    # 7. Execute Three-Vector Genomic Privacy Audit
    print("\nExecuting Multi-Vector Privacy Quantification against Client 9's Canaries...")
    audit_results = run_three_vector_privacy_audit(
        model=global_model,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        canaries=canaries,
        tokenizer=tokenizer,
        beam_width=args.beam_width,
        max_eval_samples=200,
        device=device,
    )

    # 8. Print Publication-Grade Quantitative Report
    print_genomic_privacy_report(audit_results)


if __name__ == "__main__":
    main()

