"""
OPA package - One-Shot Private Aggregation for Genomic Privacy Auditing.
"""

from .model import SimpleDNALM, TransformerBlock, CausalSelfAttention
from .dataset import (
    DNATokenizer,
    CanaryMetadata,
    GenomicDataset,
    sample_zero_order_dna,
    generate_canary_sequences,
    create_genomic_corpus,
    get_genomic_dataloaders,
    NUCLEOTIDES,
)
from .attack import (
    run_sequence_extraction_attack,
    run_perplexity_detection_attack,
    run_membership_inference_attack,
    run_three_vector_privacy_audit,
    print_genomic_privacy_report,
    compute_sequence_perplexity,
    beam_search_canary_extraction,
    compute_auc_roc,
)
from .crypto import (
    LWR_PRG,
    share,
    reconstruct,
    quantize_to_zp,
    dequantize_from_zp,
    DEFAULT_Q,
    DEFAULT_P,
    DEFAULT_SCALE,
)
from .opa_flwr import OPAClient, OPAFedAvg, OPAStrategy

__all__ = [
    "SimpleDNALM",
    "TransformerBlock",
    "CausalSelfAttention",
    "DNATokenizer",
    "CanaryMetadata",
    "GenomicDataset",
    "sample_zero_order_dna",
    "generate_canary_sequences",
    "create_genomic_corpus",
    "get_genomic_dataloaders",
    "NUCLEOTIDES",
    "run_sequence_extraction_attack",
    "run_perplexity_detection_attack",
    "run_membership_inference_attack",
    "run_three_vector_privacy_audit",
    "print_genomic_privacy_report",
    "compute_sequence_perplexity",
    "beam_search_canary_extraction",
    "compute_auc_roc",
    "LWR_PRG",
    "share",
    "reconstruct",
    "quantize_to_zp",
    "dequantize_from_zp",
    "DEFAULT_Q",
    "DEFAULT_P",
    "DEFAULT_SCALE",
    "OPAClient",
    "OPAFedAvg",
    "OPAStrategy",
]

