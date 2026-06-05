from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    recall = sensitivity
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = float(np.nanmean([sensitivity, specificity]))

    if len(np.unique(y_true)) < 2 or roc_auc_score is None:
        auroc = float("nan")
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auroc = float("nan")

    if len(np.unique(y_true)) < 2 or average_precision_score is None:
        auprc = float("nan")
    else:
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except Exception:
            auprc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "auprc": auprc,
        **cm,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "balanced_accuracy",
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_prob) == 0:
        return 0.5
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob]))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        metrics = compute_binary_metrics(y_true, y_prob, float(threshold))
        score = metrics.get(objective, float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold

