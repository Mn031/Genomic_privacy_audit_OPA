"""
dataset.py - Real Biological Genomic Dataset (GUE Benchmark) & Canary Injection (OPA).
Based on "Quantifying Memorization and Privacy Risks in Genomic Language Models" (arXiv:2603.08913v3).

Key Components:
  1. Real Biological Dataset (GUE Benchmark):
       - Fetches the Genomic Understanding Evaluation (GUE) benchmark dataset,
         specifically the `prom_300_all` promoter detection subset, using Hugging Face `datasets`.
       - Automatic Hugging Face Authentication: checks `HF_TOKEN` environment variable or
         reads `hugging_face_token.txt` if present.
  2. Sequence Processing:
       - Center-crops or pads GUE sequences to exactly 256 nucleotides.
       - Standardizes characters to uppercase.
  3. Character-Level Tokenizer (8 Tokens):
       - <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, A=4, C=5, G=6, T=7.
  4. Sequence Splits:
       - 1,000 training sequences and 200 held-out test sequences.
  5. Tiered Canary Injection (Client 9 Exclusive):
       - Generates 100 distinct 64-nt synthetic canaries (50% GC).
       - Stratified across 4 repetition tiers (1, 5, 10, 20 duplicates; 25 canaries/tier; 900 total copies).
       - Injected uniformly at random ONLY into the training split of client_9.
       - Clients 0-8 and the test split remain 100% clean real GUE biological sequences.
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
# 1. HUGGING FACE AUTHENTICATION & DATASET LOADER
# =====================================================================

def get_huggingface_token() -> Optional[str]:
    """
    Retrieves the Hugging Face authentication token:
      1. Checks the HF_TOKEN environment variable.
      2. If not set, checks for a hugging_face_token.txt file in the workspace or parent directory.
      3. Returns the extracted token string or None.
    """
    # 1. Check environment variable
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        token = token.strip()
        if "HF_TOKEN=" in token:
            token = token.split("HF_TOKEN=", 1)[1].strip()
        return token.strip(' "\'')

    # 2. Search for hugging_face_token.txt in workspace or parent directories
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ]
    for d in search_dirs:
        token_path = os.path.join(d, "hugging_face_token.txt")
        if os.path.isfile(token_path):
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if "HF_TOKEN=" in content:
                    content = content.split("HF_TOKEN=", 1)[1].strip()
                token = content.strip(' "\' \r\n')
                if token:
                    return token
            except Exception:
                pass

    return None


def load_gue_dataset_sequences(token: Optional[str] = None) -> List[str]:
    """
    Loads raw DNA sequences from the Genomic Understanding Evaluation (GUE) benchmark,
    specifically the `prom_300_all` subset via Hugging Face `datasets`.
    
    Includes robust authentication error handling instructing the user on how to set HF_TOKEN.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The Hugging Face 'datasets' library is required to load the GUE benchmark.\n"
            "Please install it using: pip install datasets"
        ) from e

    if token is None:
        token = get_huggingface_token()

    repo_candidates = [
        ("leannmlindsey/GUE", "prom_300_all"),
        ("zhihan1996/GUE", "prom_300_all"),
    ]

    raw_dataset = None
    last_error = None

    for repo, subset in repo_candidates:
        try:
            # Pass token if available; some mirrors are gated or require authenticated read
            raw_dataset = load_dataset(repo, name=subset, token=token)
            if raw_dataset is not None:
                break
        except Exception as e:
            last_error = e
            continue

    if raw_dataset is None:
        token_status = "DETECTED" if token else "MISSING / NOT SET"
        error_msg = (
            f"\n" + "=" * 80 + "\n"
            f"[ERROR] Failed to load GUE 'prom_300_all' dataset from Hugging Face.\n"
            f"Underlying error: {last_error}\n"
            f"Current HF_TOKEN status: {token_status}\n\n"
            f"HUGGING FACE AUTHENTICATION GUIDE:\n"
            f"  1. Export your Hugging Face user access token before running:\n"
            f"     In PowerShell : $env:HF_TOKEN = \"your_hf_token_here\"\n"
            f"     In Bash / Zsh : export HF_TOKEN=\"your_hf_token_here\"\n"
            f"  2. Or save your token inside 'hugging_face_token.txt' at the repository root.\n"
            f"=" * 80 + "\n"
        )
        raise RuntimeError(error_msg)

    # Extract sequences from all available dataset splits
    raw_sequences: List[str] = []
    for split_name in raw_dataset.keys():
        split_data = raw_dataset[split_name]
        for row in split_data:
            # GUE datasets typically store DNA strings in 'sequence' or 'text'
            seq = row.get("sequence") or row.get("text") or row.get("dna")
            if seq and isinstance(seq, str) and len(seq.strip()) > 0:
                raw_sequences.append(seq.strip())

    if len(raw_sequences) < 1200:
        raise ValueError(
            f"GUE prom_300_all dataset returned only {len(raw_sequences)} sequences, but at least 1,200 are required."
        )

    return raw_sequences


def process_sequence_to_length(sequence: str, target_length: int = 256) -> str:
    """
    Standardizes sequence to exactly target_length (256 nt):
      - If longer than target_length: center-crops the sequence.
      - If shorter than target_length: symmetrically pads with 'N' (mapped to <UNK>).
      - Converts all characters to uppercase.
    """
    seq = sequence.strip().upper()
    cur_len = len(seq)
    if cur_len == target_length:
        return seq
    elif cur_len > target_length:
        start = (cur_len - target_length) // 2
        return seq[start : start + target_length]
    else:
        pad_needed = target_length - cur_len
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        return ("N" * left_pad) + seq + ("N" * right_pad)


# =====================================================================
# 2. CHARACTER-LEVEL GENOMIC TOKENIZER (8 TOKENS)
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
# 3. CANARY METADATA & REPETITION TIERS
# =====================================================================

@dataclass
class CanaryMetadata:
    """Stores metadata for a confidential canary sequence."""
    canary_id: int
    sequence: str          # 64-nt string
    tier: int              # Repetition frequency: 1, 5, 10, or 20
    prefix: str            # First 32 nucleotides (prompt for extraction attack)
    target_suffix: str     # Remaining 32 nucleotides (ground truth recovery target)
    token_ids: List[int]   # Tokenized 64-nt IDs


def generate_canary_sequences(
    num_canaries: int = 100,
    canary_length: int = 64,
    tokenizer: Optional[DNATokenizer] = None,
    seed: int = 1337,
) -> List[CanaryMetadata]:
    """
    Generates 100 distinct 64-nucleotide synthetic canaries sampled with balanced 50% GC content:
      P(A) = 0.25, P(T) = 0.25, P(C) = 0.25, P(G) = 0.25.
    Assigned across 4 repetition tiers (1x, 5x, 10x, 20x duplicates; 25 canaries per tier).
    """
    if tokenizer is None:
        tokenizer = DNATokenizer()

    rng = np.random.RandomState(seed)
    tier_assignments = [1] * 25 + [5] * 25 + [10] * 25 + [20] * 25

    canaries: List[CanaryMetadata] = []
    seen_sequences = set()

    for i in range(num_canaries):
        while True:
            chars = rng.choice(NUCLEOTIDES, size=canary_length, p=[0.25, 0.25, 0.25, 0.25])
            seq = "".join(chars)
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


# =====================================================================
# 4. REAL GUE CORPUS CREATION & CANARY INJECTION
# =====================================================================

def create_genomic_corpus(
    num_train: int = 1000,
    num_test: int = 200,
    num_clients: int = 10,
    target_client_id: int = 9,
    seq_length: int = 256,
    canary_length: int = 64,
    seed: int = 42,
    hf_token: Optional[str] = None,
) -> Tuple[List[str], List[str], List[CanaryMetadata], Dict[int, List[str]]]:
    """
    Loads real biological sequences from the GUE (Genomic Understanding Evaluation) benchmark
    prom_300_all subset via Hugging Face datasets:
      1. Fetches GUE prom_300_all sequences using HF authentication token (HF_TOKEN or hugging_face_token.txt).
      2. Center-crops or pads all sequences to exactly 256 nucleotides.
      3. Shuffles deterministically with seed and partitions into:
           - 900 clean background GUE sequences for Clients 0-8 (100 seqs/client).
           - 100 clean background GUE sequences for Client 9.
           - 900 GUE sequences for Client 9 hosting the 100 canaries across 4 tiers (1x, 5x, 10x, 20x).
           - 200 held-out test GUE sequences (zero canaries).
      4. Clients 0-8 and the test split remain 100% clean baseline biological sequences.
    """
    rng = np.random.RandomState(seed)
    tokenizer = DNATokenizer()

    # 1. Fetch real biological sequences from GUE prom_300_all
    raw_gue_seqs = load_gue_dataset_sequences(token=hf_token)

    # 2. Process to exact target sequence length (256 nt)
    processed_seqs = [process_sequence_to_length(s, seq_length) for s in raw_gue_seqs]

    # Shuffle deterministically
    indices = rng.permutation(len(processed_seqs))
    shuffled_seqs = [processed_seqs[i] for i in indices]

    clean_seqs_per_client = num_train // num_clients  # 100

    # Extract clean sequences for clients 0 through 9
    client_train_seqs: Dict[int, List[str]] = {}
    ptr = 0
    for cid in range(num_clients):
        client_train_seqs[cid] = shuffled_seqs[ptr : ptr + clean_seqs_per_client]
        ptr += clean_seqs_per_client

    # Extract held-out test sequences
    test_seqs = shuffled_seqs[ptr : ptr + num_test]
    ptr += num_test

    # 3. Generate 100 distinct 64-nt canaries across 4 tiers
    canaries = generate_canary_sequences(
        num_canaries=100,
        canary_length=canary_length,
        tokenizer=tokenizer,
        seed=seed + 100,
    )

    # 4. Embed canaries exclusively into client_9's partition
    max_offset = seq_length - canary_length
    canary_host_seqs: List[str] = []

    # Use remaining real GUE sequences as hosts for canaries (cycle if necessary)
    available_hosts = shuffled_seqs[ptr:]
    if len(available_hosts) < 900:
        host_pool = shuffled_seqs[:ptr]
        host_indices = rng.choice(len(host_pool), size=900, replace=True)
        available_hosts = [host_pool[i] for i in host_indices]

    host_idx = 0
    for canary in canaries:
        for _ in range(canary.tier):
            bg = available_hosts[host_idx % len(available_hosts)]
            host_idx += 1
            offset = rng.randint(0, max_offset + 1)
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
# 5. PYTORCH DATASET & DATALOADERS
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
    hf_token: Optional[str] = None,
) -> Tuple[DNATokenizer, Dict[Union[int, str], DataLoader], DataLoader, List[CanaryMetadata], List[str], List[str]]:
    """
    Constructs the real GUE genomic dataset and Flower FL client DataLoaders:
      1. Loads 1,000 baseline GUE train sequences and 200 GUE test sequences.
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
        hf_token=hf_token,
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
