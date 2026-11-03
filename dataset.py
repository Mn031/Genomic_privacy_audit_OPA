"""
dataset.py - Genomic Dataset Synthesis, Tokenization & Canary Injection (OPA).
Based on "Quantifying Memorization and Privacy Risks in Genomic Language Models" (arXiv:2603.08913v3).

Key Components:
  1. Zero-Order Genomic Synthesis: Generates independent, identically distributed nucleotides
     (A, C, G, T with balanced 50% GC content).
  2. Character-Level Tokenizer: 8-token discrete genomic vocabulary:
     <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, A=4, C=5, G=6, T=7.
  3. Sequence Length & Splits: 1,000 training sequences (256 nt) and 200 test sequences (256 nt).
  4. Canary Sequence Injection: Generates 100 distinct 64-nt synthetic canaries (50% GC) across
     4 repetition tiers (1, 5, 10, 20 duplicates; 25 canaries/tier).
     CRITICAL: Injected uniformly at random ONLY into the training split of client_9 (Clients 0-8 remain clean).
  5. Sliding Window DataLoader: Next-token prediction causal language modeling objective.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Core Nucleotides
NUCLEOTIDES = ["A", "C", "G", "T"]


# =====================================================================
# 1. CHARACTER-LEVEL GENOMIC TOKENIZER (8 TOKENS)
# =====================================================================

class DNATokenizer:
    """
    Character-level tokenizer for genomic sequences.
    
    Vocabulary:
      0: <PAD> - Padding token
      1: <UNK> - Unknown nucleotide
      2: <BOS> - Beginning of sequence
      3: <EOS> - End of sequence
      4: A     - Adenine
      5: C     - Cytosine
      6: G     - Guanine
      7: T     - Thymine
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.nucleotides = ["A", "C", "G", "T"]
        self.vocab = self.special_tokens + self.nucleotides

        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token: Dict[int, str] = {idx: token for idx, token in enumerate(self.vocab)}

        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]
        self.bos_id = self.token2id[self.BOS_TOKEN]
        self.eos_id = self.token2id[self.EOS_TOKEN]

        # Nucleotide token ID range
        self.nuc_ids = [self.token2id[n] for n in self.nucleotides]

    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """Converts DNA string to list of token IDs."""
        seq_clean = sequence.strip().upper()
        tokens = [self.token2id.get(ch, self.unk_id) for ch in seq_clean]
        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts list of token IDs back to a DNA string."""
        chars = []
        for tid in token_ids:
            if tid in self.id2token:
                tok = self.id2token[tid]
                if skip_special_tokens and tok in self.special_tokens:
                    continue
                chars.append(tok)
            else:
                if not skip_special_tokens:
                    chars.append(self.UNK_TOKEN)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.vocab)


# =====================================================================
# 2. CANARY METADATA & REPETITION TIERS
# =====================================================================

@dataclass
class CanaryMetadata:
    """Stores metadata for a synthetic confidential canary sequence."""
    canary_id: int
    sequence: str          # 64-nt string
    tier: int              # Repetition frequency: 1, 5, 10, or 20
    prefix: str            # First 32 nucleotides (prompt for extraction attack)
    target_suffix: str     # Remaining 32 nucleotides (ground truth recovery target)
    token_ids: List[int]   # Tokenized 64-nt IDs


# =====================================================================
# 3. SYNTHETIC ZERO-ORDER CORPUS GENERATOR & CANARY INJECTOR
# =====================================================================

def sample_zero_order_dna(length: int, rng: np.random.RandomState) -> str:
    """
    Samples an independent, identically distributed DNA sequence with balanced 50% GC content:
      P(A) = 0.25, P(T) = 0.25, P(C) = 0.25, P(G) = 0.25
    """
    chars = rng.choice(NUCLEOTIDES, size=length, p=[0.25, 0.25, 0.25, 0.25])
    return "".join(chars)


def generate_canary_sequences(
    num_canaries: int = 100,
    canary_length: int = 64,
    tokenizer: Optional[DNATokenizer] = None,
    seed: int = 1337,
) -> List[CanaryMetadata]:
    """
    Generates 100 distinct 64-nucleotide canaries sampled with balanced 50% GC content.
    Assigned across 4 repetition tiers (1x, 5x, 10x, 20x duplicates; 25 canaries per tier).
    """
    if tokenizer is None:
        tokenizer = DNATokenizer()

    rng = np.random.RandomState(seed)
    tier_assignments = [1] * 25 + [5] * 25 + [10] * 25 + [20] * 25

    canaries: List[CanaryMetadata] = []
    seen_sequences = set()

    for i in range(num_canaries):
        # Ensure canary uniqueness
        while True:
            seq = sample_zero_order_dna(canary_length, rng)
            if seq not in seen_sequences:
                seen_sequences.add(seq)
                break

        prefix = seq[:32]
        target_suffix = seq[32:]
        token_ids = tokenizer.encode(seq)

        canary = CanaryMetadata(
            canary_id=i,
            sequence=seq,
            tier=tier_assignments[i],
            prefix=prefix,
            target_suffix=target_suffix,
            token_ids=token_ids,
        )
        canaries.append(canary)

    return canaries


def create_genomic_corpus(
    num_train: int = 1000,
    num_test: int = 200,
    num_clients: int = 10,
    target_client_id: int = 9,
    seq_length: int = 256,
    canary_length: int = 64,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[CanaryMetadata], Dict[int, List[str]]]:
    """
    Generates the complete synthetic genomic dataset:
      - Generates clean zero-order sequences (100 per client across 10 clients).
      - Generates 200 held-out test sequences (256 nt each).
      - Generates 100 distinct 64-nt canaries across 4 repetition tiers (1x, 5x, 10x, 20x duplicates).
      - Injects all 900 canary copies uniformly at random ONLY into the training split of client_9,
        embedding each canary into a 256-nt host sequence without overwriting or collisions.
      - Clients 0 through 8 remain 100% clean background sequences (zero canaries).
      - Test split never receives any canaries.
    """
    rng = np.random.RandomState(seed)
    tokenizer = DNATokenizer()

    # 1. Clean background sequences per client
    seqs_per_client = num_train // num_clients
    client_train_seqs: Dict[int, List[str]] = {}
    for cid in range(num_clients):
        client_train_seqs[cid] = [sample_zero_order_dna(seq_length, rng) for _ in range(seqs_per_client)]

    test_seqs = [sample_zero_order_dna(seq_length, rng) for _ in range(num_test)]

    # 2. Generate 100 distinct 64-nt canaries across 4 tiers
    canaries = generate_canary_sequences(
        num_canaries=100,
        canary_length=canary_length,
        tokenizer=tokenizer,
        seed=seed + 100,
    )

    # 3. Embed canaries exclusively into client_9's partition
    max_offset = seq_length - canary_length
    canary_host_seqs: List[str] = []

    for canary in canaries:
        for _ in range(canary.tier):
            # Generate a 256-nt host sequence with balanced zero-order background
            bg = sample_zero_order_dna(seq_length, rng)
            offset = rng.randint(0, max_offset + 1)
            # Embed 64-nt canary at random offset
            embedded = bg[:offset] + canary.sequence + bg[offset + canary_length:]
            canary_host_seqs.append(embedded)

    # Shuffle canary host sequences and append to client_9
    rng.shuffle(canary_host_seqs)
    client_train_seqs[target_client_id].extend(canary_host_seqs)

    # Flatten train_seqs for global evaluation
    train_seqs: List[str] = []
    for cid in range(num_clients):
        train_seqs.extend(client_train_seqs[cid])

    return train_seqs, test_seqs, canaries, client_train_seqs


# =====================================================================
# 4. PYTORCH DATASET & DATALOADERS
# =====================================================================

class GenomicDataset(Dataset):
    """
    PyTorch Dataset for next-token causal genomic language modeling.
    For an input token sequence x = [t_0, ..., t_{L-2}], target is y = [t_1, ..., t_{L-1}].
    """
    def __init__(self, sequences: List[str], tokenizer: DNATokenizer) -> None:
        self.tokenizer = tokenizer
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

        for seq in sequences:
            token_ids = tokenizer.encode(seq)
            if len(token_ids) >= 2:
                self.inputs.append(torch.tensor(token_ids[:-1], dtype=torch.long))
                self.targets.append(torch.tensor(token_ids[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def get_genomic_dataloaders(
    num_clients: int = 10,
    num_train: int = 1000,
    num_test: int = 200,
    seq_length: int = 256,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[DNATokenizer, Dict[Union[int, str], DataLoader], DataLoader, List[CanaryMetadata], List[str], List[str]]:
    """
    Constructs the complete genomic dataset and Flower FL client DataLoaders:
      1. Generates 1,000 baseline train sequences and 200 test sequences.
      2. Injects all 100 canaries across 4 tiers (900 copies) exclusively into Client 9.
      3. Constructs PyTorch DataLoaders for each client and the held-out test split.
      
    Returns:
        (tokenizer, client_loaders, test_loader, canaries, train_seqs, test_seqs)
    """
    tokenizer = DNATokenizer()

    train_seqs, test_seqs, canaries, client_train_seqs = create_genomic_corpus(
        num_train=num_train,
        num_test=num_test,
        num_clients=num_clients,
        target_client_id=9,
        seq_length=seq_length,
        canary_length=64,
        seed=seed,
    )

    client_loaders: Dict[Union[int, str], DataLoader] = {}
    for cid in range(num_clients):
        c_seqs = client_train_seqs[cid]
        c_dataset = GenomicDataset(c_seqs, tokenizer=tokenizer)
        c_loader = DataLoader(c_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders[cid] = c_loader
        client_loaders[f"client_{cid}"] = c_loader

    # Held-out test DataLoader
    test_dataset = GenomicDataset(test_seqs, tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return tokenizer, client_loaders, test_loader, canaries, train_seqs, test_seqs
