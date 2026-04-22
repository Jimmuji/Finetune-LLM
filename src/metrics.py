"""
Disease label evaluation metrics.

Primary metric: Entity-level F1
- Each comma-separated disease name is treated as one entity
- Normalize (lowercase, strip punctuation/whitespace)
- Compute Precision, Recall, F1 across entities

Secondary metrics: Jaccard similarity, Exact Match
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_label(s: str) -> str:
    """Lowercase, strip trailing punctuation and extra whitespace."""
    s = str(s).strip().lower()
    s = re.sub(r"[.;:\u3002\uff0c\uff1b]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_labels(text: str) -> Set[str]:
    """
    Parse a raw prediction/label string into a normalized set of entity strings.

    Handles comma, Chinese comma, semicolon, newline as separators.
    Returns empty set for None/empty/no-disease strings.
    """
    if not text or str(text).strip().lower() in {"none", "nil", "n/a", ""}:
        return set()
    # Strip <think>...</think> blocks that Qwen3 may emit
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    parts = re.split(r"[,，;；\n]", text)
    return {normalize_label(p) for p in parts if normalize_label(p)}


# ── Per-sample metrics ─────────────────────────────────────────────────────────

def entity_prf1(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    """Precision, Recall, F1 for two entity sets."""
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def jaccard_similarity(pred: Set[str], gold: Set[str]) -> float:
    """Jaccard index between two label sets."""
    if not pred and not gold:
        return 1.0
    return len(pred & gold) / len(pred | gold)


def exact_match(pred: Set[str], gold: Set[str]) -> float:
    """1.0 if the prediction exactly equals gold, else 0.0."""
    return 1.0 if pred == gold else 0.0


# ── Corpus-level evaluation ────────────────────────────────────────────────────

def evaluate(predictions: List[str], gold_labels: List[str]) -> Dict[str, float]:
    """
    Macro-average entity-level metrics over all samples.

    Args:
        predictions: list of raw model output strings
        gold_labels:  list of ground-truth label strings (comma-separated)

    Returns:
        dict with keys: precision, recall, f1, jaccard, exact_match, n_samples
    """
    assert len(predictions) == len(gold_labels), "Length mismatch"

    records = []
    for pred_raw, gold_raw in zip(predictions, gold_labels):
        pred = parse_labels(pred_raw)
        gold = parse_labels(gold_raw)
        p, r, f1 = entity_prf1(pred, gold)
        j  = jaccard_similarity(pred, gold)
        em = exact_match(pred, gold)
        records.append((p, r, f1, j, em))

    arr = np.array(records)
    return {
        "precision":   float(arr[:, 0].mean()),
        "recall":      float(arr[:, 1].mean()),
        "f1":          float(arr[:, 2].mean()),
        "jaccard":     float(arr[:, 3].mean()),
        "exact_match": float(arr[:, 4].mean()),
        "n_samples":   len(records),
    }


def print_metrics(metrics: Dict[str, float], label: str = "") -> None:
    """Print evaluation metrics in a readable table."""
    header = f"=== {label} ===" if label else "=== Evaluation Results ==="
    print(header)
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1 (macro) : {metrics['f1']:.4f}   ← primary metric")
    print(f"  Jaccard    : {metrics['jaccard']:.4f}")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  N Samples  : {metrics['n_samples']}")
