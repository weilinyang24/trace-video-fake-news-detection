from __future__ import annotations

from .labels import LABEL_ORDER, normalize_gold, normalize_prediction


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    """Compute the same acc and unweighted macro metrics as VideoMMD."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} != {len(pred)}")
    gold = [normalize_gold(item) for item in gold]
    pred = [normalize_prediction(item)[0] for item in pred]
    total = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in LABEL_ORDER:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in gold),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": _safe_div(correct, total),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "total": total,
        "per_class": per_class,
    }

