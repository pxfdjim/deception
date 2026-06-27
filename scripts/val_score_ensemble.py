import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-calibrated score ensemble")
    parser.add_argument("--dataset", choices=("dolos", "seumld"), required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--folds", default="")
    parser.add_argument("--num-random", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--f1-weight", type=float, default=0.25)
    parser.add_argument("--selection-metric", choices=("acc_f1", "f1_acc"), default="acc_f1")
    return parser.parse_args()


def metric_pack(probs, labels, threshold):
    preds = (probs > threshold).astype(np.int64)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = metrics.roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(metrics.accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(threshold),
    }


def best_threshold(probs, labels, args):
    best_key = None
    best_metrics = None
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    for threshold in thresholds:
        item = metric_pack(probs, labels, threshold)
        if args.selection_metric == "f1_acc":
            key = (item["f1"], item["accuracy"], item["auc"], -abs(threshold - 0.5))
        else:
            key = (
                item["accuracy"] + args.f1_weight * item["f1"],
                item["accuracy"],
                item["f1"],
                item["auc"],
                -abs(threshold - 0.5),
            )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = item
    return best_metrics, best_key


def candidate_weights(num_members, num_random, seed):
    weights = []
    weights.append(np.ones(num_members, dtype=np.float64) / num_members)
    for idx in range(num_members):
        item = np.zeros(num_members, dtype=np.float64)
        item[idx] = 1.0
        weights.append(item)
    if num_members == 2:
        for alpha in np.linspace(0.0, 1.0, 101):
            weights.append(np.asarray([alpha, 1.0 - alpha], dtype=np.float64))
    rng = np.random.default_rng(seed)
    for _ in range(num_random):
        weights.append(rng.dirichlet(np.ones(num_members, dtype=np.float64)))
    return weights


def load_fold(member_dirs, fold):
    val_probs = []
    test_probs = []
    val_labels = None
    test_labels = None
    for member_dir in member_dirs:
        path = member_dir / f"fold_{fold}_scores.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        if val_labels is None:
            val_labels = data["val_labels"].astype(np.int64)
            test_labels = data["test_labels"].astype(np.int64)
        else:
            if not np.array_equal(val_labels, data["val_labels"].astype(np.int64)):
                raise ValueError(f"Validation labels mismatch at {path}")
            if not np.array_equal(test_labels, data["test_labels"].astype(np.int64)):
                raise ValueError(f"Test labels mismatch at {path}")
        val_probs.append(data["val_probs"].astype(np.float64))
        test_probs.append(data["test_probs"].astype(np.float64))
    return np.stack(val_probs), val_labels, np.stack(test_probs), test_labels


def main():
    args = parse_args()
    root = Path("/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/val_calibrated") / args.dataset
    member_dirs = [root / member for member in args.members]
    out_dir = root / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(message):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    if log_path.exists():
        log_path.unlink()

    if args.folds:
        folds = [int(item) for item in args.folds.split(",") if item.strip()]
    else:
        fold_set = set()
        for member_dir in member_dirs:
            for score_path in member_dir.glob("fold_*_scores.npz"):
                fold_set.add(int(score_path.stem.split("_")[1]))
        folds = sorted(fold_set)

    log(f"dataset={args.dataset} exp={args.exp_name}")
    log(f"members={','.join(args.members)}")
    log(f"selection={args.selection_metric} f1_weight={args.f1_weight} random={args.num_random}")

    rows = []
    for fold in folds:
        val_stack, val_labels, test_stack, test_labels = load_fold(member_dirs, fold)
        best = None
        best_weights = None
        best_val_metrics = None
        for weights in candidate_weights(len(member_dirs), args.num_random, args.seed + fold):
            val_probs = np.tensordot(weights, val_stack, axes=(0, 0))
            val_metrics, key = best_threshold(val_probs, val_labels, args)
            if best is None or key > best:
                best = key
                best_weights = weights
                best_val_metrics = val_metrics

        test_probs = np.tensordot(best_weights, test_stack, axes=(0, 0))
        test_metrics = metric_pack(test_probs, test_labels, best_val_metrics["threshold"])
        row = {
            "fold": fold,
            "threshold": best_val_metrics["threshold"],
            "weights": ",".join(f"{item:.4f}" for item in best_weights),
            "val_accuracy": best_val_metrics["accuracy"],
            "val_f1": best_val_metrics["f1"],
            "val_auc": best_val_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        }
        rows.append(row)
        log(
            f"Fold {fold}: val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1']:.4f} "
            f"th={row['threshold']:.3f} weights={row['weights']} | "
            f"test_acc={row['test_accuracy']:.4f} test_f1={row['test_f1']:.4f} "
            f"test_auc={row['test_auc']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    log("\nSummary:")
    for key in ("test_accuracy", "test_f1", "test_auc", "test_precision", "test_recall"):
        log(f"{key}: {mean[key]:.4f} +/- {std[key]:.4f}")


if __name__ == "__main__":
    main()
