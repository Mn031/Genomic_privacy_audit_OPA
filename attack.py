"""
attack.py - Three-Vector Genomic Privacy Attack Suite (OPA).
Based on "Quantifying Memorization and Privacy Risks in Genomic Language Models" (arXiv:2603.08913v3).

Three Attack Vectors:
  1. Vector 1: Sequence Extraction Attack (Tab Attack via Beam Search)
     - Condition on 32-nt prefix, recover remaining 32-nt suffix using beam search (width W=10).
     - Calculate Canary Exposure: E(s) = 64 - log2(Rank(s)) bits (max: 64.0 bits).
     - Compute extraction rate s_ext = (extracted / total) across repetition tiers (1, 5, 10, 20).
  2. Vector 2: Perplexity-Based Detection Attack
     - Compute sequence perplexity across train, test, and canary sequences.
     - Calculate Memorization Gap Ratio = PP_test / PP_canary.
     - Compute normalized score s_ppl = max(0, 1 - PP_canary / PP_test).
  3. Vector 3: Membership Inference Attack (LiRA Formulation)
     - Fit parametric Gaussians N(mu_in, sigma_in^2) and N(mu_out, sigma_out^2) to train/test losses.
     - Compute Likelihood Ratio scores and evaluate AUC-ROC.
     - Rescale score: s_mia = max(0, 2 * (AUC - 0.5)).
  4. Unified Privacy Vulnerability Score:
     - S_config = max(s_ppl, s_ext, s_mia) in [0, 1].
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from OPA.dataset import DNATokenizer, CanaryMetadata
except ImportError:
    try:
        from Genomic_privacy_audit.OPA.dataset import DNATokenizer, CanaryMetadata
    except ImportError:
        from dataset import DNATokenizer, CanaryMetadata


# =====================================================================
# VECTOR 1: SEQUENCE EXTRACTION ATTACK (BEAM SEARCH ON 32-NT PREFIX)
# =====================================================================

def beam_search_canary_extraction(
    model: nn.Module,
    prefix_str: str,
    target_suffix_str: str,
    tokenizer: DNATokenizer,
    beam_width: int = 10,
    max_tokens: int = 32,
    max_candidates: int = 1000,
    device: Optional[torch.device] = None,
) -> Tuple[bool, str, float, int]:
    """
    Conditions on a 32-nucleotide prefix and uses beam search to recover the remaining 32-nt suffix.
    
    Returns:
      (success, best_generated_suffix, exposure_bits, rank)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    nuc_token_ids = [tokenizer.token2id[n] for n in ["A", "C", "G", "T"]]
    prefix_ids = tokenizer.encode(prefix_str)

    # Beam item: (log_prob, token_ids)
    beams: List[Tuple[float, List[int]]] = [(0.0, prefix_ids)]
    completed_hypotheses: List[Tuple[float, str]] = []

    for step in range(max_tokens):
        candidates: List[Tuple[float, List[int]]] = []

        # Batched forward pass across all active beam hypotheses
        inp_tensor = torch.tensor([seq_ids for _, seq_ids in beams], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(inp_tensor)
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

        for b_idx, (log_prob, seq_ids) in enumerate(beams):
            for token_id in nuc_token_ids:
                cand_log_prob = log_prob + float(log_probs[b_idx, token_id].item())
                cand_ids = seq_ids + [token_id]
                candidates.append((cand_log_prob, cand_ids))

        # Prune to top beam_width hypotheses
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        if len(completed_hypotheses) < max_candidates:
            for lp, ids in candidates:
                suffix = tokenizer.decode(ids[len(prefix_ids):])
                completed_hypotheses.append((lp, suffix))

    best_log_prob, best_ids = beams[0]
    best_suffix = tokenizer.decode(best_ids[len(prefix_ids):])

    # Deduplicate hypotheses by highest score
    unique_hypotheses: Dict[str, float] = {}
    for lp, hyp_suffix in completed_hypotheses:
        if hyp_suffix not in unique_hypotheses or lp > unique_hypotheses[hyp_suffix]:
            unique_hypotheses[hyp_suffix] = lp

    sorted_hypotheses = sorted(unique_hypotheses.items(), key=lambda x: x[1], reverse=True)

    rank = max_candidates + 1
    for r, (hyp_suffix, _) in enumerate(sorted_hypotheses, 1):
        if hyp_suffix == target_suffix_str:
            rank = r
            break

    # Exposure metric: E(s) = 64 - log2(rank)
    if rank <= max_candidates:
        exposure = max(0.0, 64.0 - math.log2(rank))
    else:
        exposure = 0.0

    success = (best_suffix == target_suffix_str)
    return success, best_suffix, exposure, rank


def run_sequence_extraction_attack(
    model: nn.Module,
    canaries: List[CanaryMetadata],
    tokenizer: DNATokenizer,
    beam_width: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, Dict]]:
    """Runs Vector 1 (Sequence Extraction Attack) across all 100 canaries."""
    total_canaries = len(canaries)
    extracted_count = 0
    total_exposure = 0.0

    tier_stats: Dict[int, Dict[str, Union[int, float]]] = {
        1: {"total": 0, "extracted": 0, "sum_exposure": 0.0},
        5: {"total": 0, "extracted": 0, "sum_exposure": 0.0},
        10: {"total": 0, "extracted": 0, "sum_exposure": 0.0},
        20: {"total": 0, "extracted": 0, "sum_exposure": 0.0},
    }

    details = []

    for canary in canaries:
        success, generated_suffix, exposure, rank = beam_search_canary_extraction(
            model=model,
            prefix_str=canary.prefix,
            target_suffix_str=canary.target_suffix,
            tokenizer=tokenizer,
            beam_width=beam_width,
            max_tokens=32,
            max_candidates=1000,
            device=device,
        )

        if success:
            extracted_count += 1
        total_exposure += exposure

        tier = canary.tier
        if tier in tier_stats:
            tier_stats[tier]["total"] += 1
            if success:
                tier_stats[tier]["extracted"] += 1
            tier_stats[tier]["sum_exposure"] += exposure

        details.append({
            "canary_id": canary.canary_id,
            "tier": tier,
            "success": success,
            "prefix": canary.prefix,
            "target": canary.target_suffix,
            "generated": generated_suffix,
            "exposure": exposure,
            "rank": rank,
        })

    s_ext = float(extracted_count / max(total_canaries, 1))
    mean_exposure = float(total_exposure / max(total_canaries, 1))

    tier_summary = {}
    for tier, data in sorted(tier_stats.items()):
        cnt = data["total"]
        ext = data["extracted"]
        exp = data["sum_exposure"]
        tier_summary[tier] = {
            "total": cnt,
            "extracted": ext,
            "success_rate": float(ext / max(cnt, 1)),
            "mean_exposure": float(exp / max(cnt, 1)),
        }

    return {
        "s_ext": s_ext,
        "mean_exposure": mean_exposure,
        "total_extracted": extracted_count,
        "total_canaries": total_canaries,
        "tier_summary": tier_summary,
        "details": details,
    }


# =====================================================================
# VECTOR 2: PERPLEXITY DETECTION ATTACK
# =====================================================================

def compute_sequence_perplexity(
    model: nn.Module,
    sequence: str,
    tokenizer: DNATokenizer,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Computes perplexity (PP) and mean cross-entropy loss (NLL) for a DNA sequence."""
    if device is None:
        device = next(model.parameters()).device

    token_ids = tokenizer.encode(sequence)
    if len(token_ids) < 2:
        return float("inf"), float("inf")

    inp_t = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    tgt_t = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out_logits, _ = model(inp_t)
        loss = F.cross_entropy(out_logits.reshape(-1, model.vocab_size), tgt_t.reshape(-1), ignore_index=0)

    nll = float(loss.item())
    ppl = float(math.exp(min(nll, 100.0)))
    return ppl, nll


def run_perplexity_detection_attack(
    model: nn.Module,
    train_seqs: List[str],
    test_seqs: List[str],
    canaries: List[CanaryMetadata],
    tokenizer: DNATokenizer,
    max_eval_samples: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, Dict]]:
    """Evaluates memorization via perplexity distributions: PP_train, PP_test, PP_canary."""
    eval_train = train_seqs[:max_eval_samples]
    eval_test = test_seqs[:max_eval_samples]

    train_ppls = [compute_sequence_perplexity(model, s, tokenizer, device)[0] for s in eval_train]
    test_ppls = [compute_sequence_perplexity(model, s, tokenizer, device)[0] for s in eval_test]
    canary_ppls = [compute_sequence_perplexity(model, c.sequence, tokenizer, device)[0] for c in canaries]

    mean_train_ppl = float(np.mean(train_ppls))
    mean_test_ppl = float(np.mean(test_ppls))
    mean_canary_ppl = float(np.mean(canary_ppls))

    # Memorization Gap Ratio = PP_test / PP_canary
    memorization_gap = float(mean_test_ppl / max(mean_canary_ppl, 1e-8))

    # Normalized score: s_ppl = max(0, 1 - PP_canary / PP_test)
    ratio = mean_canary_ppl / max(mean_test_ppl, 1e-8)
    s_ppl = float(max(0.0, min(1.0, 1.0 - ratio)))

    return {
        "s_ppl": s_ppl,
        "memorization_gap": memorization_gap,
        "mean_train_ppl": mean_train_ppl,
        "mean_test_ppl": mean_test_ppl,
        "mean_canary_ppl": mean_canary_ppl,
    }


# =====================================================================
# VECTOR 3: MEMBERSHIP INFERENCE ATTACK (LiRA)
# =====================================================================

def compute_auc_roc(y_true: List[int], scores: List[float]) -> float:
    """Computes AUC-ROC score with an exact Wilcoxon-Mann-Whitney NumPy fallback."""
    if HAS_SKLEARN:
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            pass

    y_arr = np.array(y_true)
    s_arr = np.array(scores)
    pos_scores = s_arr[y_arr == 1]
    neg_scores = s_arr[y_arr == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count pairs where pos > neg (ties count 0.5)
    diff = pos_scores[:, None] - neg_scores[None, :]
    wins = np.sum(diff > 0)
    ties = np.sum(diff == 0)
    return float((wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores)))


def run_membership_inference_attack(
    model: nn.Module,
    train_seqs: List[str],
    test_seqs: List[str],
    tokenizer: DNATokenizer,
    max_eval_samples: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Executes Likelihood-Ratio Membership Inference Attack (LiRA)."""
    if device is None:
        device = next(model.parameters()).device

    eval_train = train_seqs[:max_eval_samples]
    eval_test = test_seqs[:max_eval_samples]

    def get_losses(seq_list: List[str]) -> List[float]:
        losses = []
        for s in seq_list:
            t_ids = tokenizer.encode(s)
            if len(t_ids) < 2:
                continue
            inp = torch.tensor([t_ids[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([t_ids[1:]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), tgt.reshape(-1), ignore_index=0)
                losses.append(float(loss.item()))
        return losses

    train_losses = get_losses(eval_train)
    test_losses = get_losses(eval_test)

    # Parametric Gaussian fit to loss distributions
    mu_in = float(np.mean(train_losses))
    sigma_in = float(np.std(train_losses) + 1e-8)

    mu_out = float(np.mean(test_losses))
    sigma_out = float(np.std(test_losses) + 1e-8)

    def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
        return -0.5 * math.log(2 * math.pi) - math.log(sigma) - ((x - mu) ** 2) / (2 * (sigma ** 2))

    y_true = [1] * len(train_losses) + [0] * len(test_losses)
    all_losses = train_losses + test_losses

    # Likelihood Ratio scores: Lambda(x) = log P(loss | in) - log P(loss | out)
    lr_scores = [log_normal_pdf(l, mu_in, sigma_in) - log_normal_pdf(l, mu_out, sigma_out) for l in all_losses]

    auc = compute_auc_roc(y_true, lr_scores)
    # Rescaled MIA vulnerability: s_mia = max(0, 2 * (AUC - 0.5))
    s_mia = float(max(0.0, min(1.0, 2.0 * (auc - 0.5))))

    return {
        "s_mia": s_mia,
        "auc_roc": auc,
        "mu_in": mu_in,
        "sigma_in": sigma_in,
        "mu_out": mu_out,
        "sigma_out": sigma_out,
    }


# =====================================================================
# UNIFIED PRIVACY AUDIT & REPORTING
# =====================================================================

def run_three_vector_privacy_audit(
    model: nn.Module,
    train_seqs: List[str],
    test_seqs: List[str],
    canaries: List[CanaryMetadata],
    tokenizer: DNATokenizer,
    beam_width: int = 10,
    max_eval_samples: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, Dict]]:
    """Executes the complete Three-Vector Privacy Attack Framework."""
    print("\n" + "=" * 80)
    print(" EXECUTING THREE-VECTOR GENOMIC PRIVACY AUDIT (arXiv:2603.08913v3)")
    print("=" * 80)

    # 1. Vector 1: Sequence Extraction Attack
    print("[Vector 1/3] Running Sequence Extraction Attack (Beam Search, width=10)...")
    v1_results = run_sequence_extraction_attack(
        model=model,
        canaries=canaries,
        tokenizer=tokenizer,
        beam_width=beam_width,
        device=device,
    )
    s_ext = v1_results["s_ext"]
    print(f"    -> Sequence Extraction Success Rate (s_ext): {s_ext * 100:.2f}%")
    print(f"    -> Mean Canary Exposure                    : {v1_results['mean_exposure']:.2f} bits")

    # 2. Vector 2: Perplexity-Based Detection
    print("\n[Vector 2/3] Running Perplexity-Based Detection Attack...")
    v2_results = run_perplexity_detection_attack(
        model=model,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        canaries=canaries,
        tokenizer=tokenizer,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    s_ppl = v2_results["s_ppl"]
    print(f"    -> Memorization Gap Ratio (Test/Canary)    : {v2_results['memorization_gap']:.2f}x")
    print(f"    -> Normalized Perplexity Score (s_ppl)     : {s_ppl:.4f}")

    # 3. Vector 3: Membership Inference Attack
    print("\n[Vector 3/3] Running Membership Inference Attack (LiRA Likelihood Ratio)...")
    v3_results = run_membership_inference_attack(
        model=model,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        tokenizer=tokenizer,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    s_mia = v3_results["s_mia"]
    print(f"    -> Membership Inference AUC-ROC            : {v3_results['auc_roc']:.4f}")
    print(f"    -> Normalized MIA Score (s_mia)            : {s_mia:.4f}")

    # 4. Maximum Vulnerability Score S_config = max(s_ppl, s_ext, s_mia)
    s_config = float(max(s_ppl, s_ext, s_mia))

    return {
        "s_config": s_config,
        "s_ext": s_ext,
        "s_ppl": s_ppl,
        "s_mia": s_mia,
        "vector_1": v1_results,
        "vector_2": v2_results,
        "vector_3": v3_results,
    }


def print_genomic_privacy_report(audit_results: Dict[str, Union[float, Dict]]) -> None:
    """Prints a high-visibility, publication-grade quantitative privacy audit report."""
    v1 = audit_results["vector_1"]
    v2 = audit_results["vector_2"]
    v3 = audit_results["vector_3"]

    s_config = audit_results["s_config"]
    s_ext = audit_results["s_ext"]
    s_ppl = audit_results["s_ppl"]
    s_mia = audit_results["s_mia"]

    print("\n" + "=" * 90)
    print("      GENOMIC PRIVACY AUDIT: MULTI-VECTOR QUANTIFICATION (arXiv:2603.08913v3)")
    print("=" * 90)
    print(f"Maximum Vulnerability Score S_config = max(s_ppl, s_ext, s_mia) : {s_config:.4f}")
    print(f"  * Vector 1 (Sequence Extraction Rate s_ext)                 : {s_ext:.4f} ({s_ext * 100:.1f}%)")
    print(f"  * Vector 2 (Normalized Perplexity Score s_ppl)               : {s_ppl:.4f}")
    print(f"  * Vector 3 (Rescaled Membership Inference Score s_mia)       : {s_mia:.4f} (AUC: {v3['auc_roc']:.4f})")
    print("-" * 90)

    # 1. Tier Breakdown Table
    print("\n [VECTOR 1] CANARY EXTRACTION SUCCESS BY REPETITION TIER (Target: Client 9):")
    print(f"{'Tier':<10} | {'Duplicates':<12} | {'Canaries':<10} | {'Extracted':<11} | {'Success Rate':<14} | {'Mean Exposure'}")
    print("-" * 90)
    for tier, data in sorted(v1["tier_summary"].items()):
        print(
            f"Tier {tier:<5} | "
            f"{tier:<12} | "
            f"{data['total']:<10} | "
            f"{data['extracted']:<11} | "
            f"{data['success_rate'] * 100:>6.1f}%       | "
            f"{data['mean_exposure']:>5.2f} / 64 bits"
        )
    print(f"{'Overall':<10} | {'--':<12} | {v1['total_canaries']:<10} | {v1['total_extracted']:<11} | {s_ext * 100:>6.1f}%       | {v1['mean_exposure']:>5.2f} / 64 bits")

    # 2. Perplexity Breakdown
    print("\n [VECTOR 2] PERPLEXITY-BASED DETECTION & MEMORIZATION GAP:")
    print(f"  - Training Population Perplexity (PP_train) : {v2['mean_train_ppl']:.2f}")
    print(f"  - Held-out Test Perplexity       (PP_test)  : {v2['mean_test_ppl']:.2f}")
    print(f"  - Canary Population Perplexity   (PP_canary): {v2['mean_canary_ppl']:.2f}")
    print(f"  - Memorization Gap Ratio (PP_test / PP_canary): {v2['memorization_gap']:.2f}x")
    print(f"  - Normalized Perplexity Score s_ppl        : {s_ppl:.4f}")

    # 3. Membership Inference Breakdown
    print("\n [VECTOR 3] MEMBERSHIP INFERENCE ATTACK (LiRA):")
    print(f"  - Member Loss Fit     : N(mu={v3['mu_in']:.4f}, sigma={v3['sigma_in']:.4f})")
    print(f"  - Non-Member Loss Fit : N(mu={v3['mu_out']:.4f}, sigma={v3['sigma_out']:.4f})")
    print(f"  - Likelihood Ratio AUC-ROC                 : {v3['auc_roc']:.4f}")
    print(f"  - Rescaled MIA Score s_mia = 2*(AUC - 0.5)  : {s_mia:.4f}")

    print("\n" + "=" * 90)
    print(" Mathematical Formulation:")
    print(" -> S_config = max(s_ppl, s_ext, s_mia) unifies privacy risk into a single worst-case")
    print("    metric bounded in [0, 1] sharing a common zero point across all threat vectors.")
    print("=" * 90 + "\n")

