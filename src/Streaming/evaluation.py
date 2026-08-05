import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calc_sigma_mad(buf: np.ndarray, eps: float = 1e-3) -> float:
    # eps is a sigma floor in the same units as buf
    med = np.median(buf)
    mad = np.median(np.abs(buf - med))
    sigma = 1.4826 * mad
    return float(max(sigma, eps))

def calc_sigma_iqr(buf: np.ndarray, eps: float) -> float:
    q25, q75 = np.quantile(buf, [0.25, 0.75])
    sigma = (q75 - q25) / 1.349  # ~std if normal
    return float(max(sigma, eps))

def extract_stats(y, bucket, name):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(bucket.get("scores_0_1", []), dtype=float)

    if len(scores) != len(y):
        raise ValueError(f"Length mismatch for {name}: scores={len(scores)} labels={len(y)}")

    # filter non-finite scores (optional but robust)
    finite = np.isfinite(scores)
    y2 = y[finite]
    s2 = scores[finite]

    # If nothing left, return NaNs
    if len(s2) == 0:
        conf_pr = float("nan")
        conf_roc = float("nan")
    else:
        # PR/AUC undefined if only one class present
        if len(np.unique(y2)) < 2:
            conf_pr = float("nan")
            conf_roc = float("nan")
        else:
            conf_pr = float(average_precision_score(y2, s2))
            conf_roc = float(roc_auc_score(y2, s2))

    tp = int(bucket.get("tp", 0))
    fp = int(bucket.get("fp", 0))
    tn = int(bucket.get("tn", 0))
    fn = int(bucket.get("fn", 0))

    prec = tp/(tp+fp) if (tp+fp) > 0 else float("nan")
    rec  = tp/(tp+fn) if (tp+fn) > 0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if np.isfinite(prec) and np.isfinite(rec) and (prec+rec) > 0 else float("nan")
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "conf_pr": conf_pr,
        "conf_roc": conf_roc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "score_mean": float(np.nanmean(s2)) if len(s2) > 0 else float("nan"),
        "score_std": float(np.nanstd(s2)) if len(s2) > 0 else float("nan"),
        "n": int(len(scores)),
        "n_finite": int(np.sum(finite)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    }

def pick_best_threshold(y_true: np.ndarray, y_score: np.ndarray, grid=None,
                        objective: str = "f1", beta: float = 2.0,
                        min_precision: float = 0.0, min_recall: float = 0.0):
    """
    Sweep a threshold grid and pick the best one according to `objective`.

    objective options:
      'f1'                  — maximize F1 (balanced precision/recall)
      'fbeta'               — maximize Fβ with `beta`; β>1 favours recall
                              (β=2 is recommended for anomaly detection with class imbalance)
      'recall_at_precision' — maximize recall subject to precision ≥ min_precision
      'precision_at_recall' — maximize precision subject to recall   ≥ min_recall

    Every stats dict always contains both f1 and f2 (β=2) so both are logged
    regardless of which objective was optimized.

    Returns: best_thr, best_stats_dict, all_stats_list
    """
    if objective not in ("f1", "fbeta", "recall_at_precision", "precision_at_recall", "youden"):
        raise ValueError(f"Unknown objective {objective!r}. "
                         "Choose from: f1, fbeta, recall_at_precision, precision_at_recall, youden")

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # remove non-finite scores (keep labels aligned)
    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]

    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    best_thr = None
    best_val = -1.0
    best_stats = None
    all_stats = []

    for thr in grid:
        y_hat = (y_score >= thr).astype(int)

        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tnr  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ok   = np.isfinite(prec) and np.isfinite(rec)
        ok_r = np.isfinite(rec)  and np.isfinite(tnr)
        f1   = (2 * prec * rec / (prec + rec))     if ok   and (prec + rec) > 0     else np.nan
        f2   = (5 * prec * rec / (4 * prec + rec)) if ok   and (4 * prec + rec) > 0 else np.nan
        acc  = (tp + tn) / max(tp + tn + fp + fn, 1)

        stats = {"thr": float(thr), "f1": float(f1), "f2": float(f2),
                 "precision": float(prec), "recall": float(rec), "accuracy": float(acc),
                 "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        all_stats.append(stats)

        if objective == "f1":
            val = f1
        elif objective == "fbeta":
            b2  = beta ** 2
            val = ((1 + b2) * prec * rec / (b2 * prec + rec)) if ok and (b2 * prec + rec) > 0 else np.nan
        elif objective == "youden":
            # J = TPR + TNR - 1; immune to class imbalance, finds the crossing point
            val = (rec + tnr - 1.0) if ok_r else np.nan
        elif objective == "recall_at_precision":
            val = rec if ok and prec >= min_precision else np.nan
        else:  # precision_at_recall
            val = prec if ok and rec >= min_recall else np.nan

        if np.isfinite(val) and val > best_val:
            best_val = val
            best_thr = float(thr)
            best_stats = stats

    # Fallback for all-negative data (no finite objective value): use thr=0.5 so
    # tp/fp/tn/fn still get logged and the confusion grid is not empty.
    if best_stats is None and all_stats:
        mid = min(all_stats, key=lambda s: abs(s["thr"] - 0.5))
        best_thr = mid["thr"]
        best_stats = mid

    return best_thr, best_stats, all_stats

def find_best_threshold_f1(y: np.ndarray, s: np.ndarray, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best = {"thr": 0.5, "f1": -1.0, "precision": np.nan, "recall": np.nan}
    for thr in grid:
        y_hat = (s >= thr).astype(int)

        tp = int(((y_hat == 1) & (y == 1)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        fn = int(((y_hat == 0) & (y == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1   = (2 * prec * rec / (prec + rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan

        if np.isfinite(f1) and f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "precision": float(prec), "recall": float(rec)}

    return best
