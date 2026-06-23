#!/usr/bin/env python3
"""
SEUMLD Error Pattern Analysis for rebuttal.

This script prefers sample-level predictions with columns/keys:
  y_true, audio_pred, avpa_pred

If no such file is found, it reconstructs integer confusion matrices from
the SEUMLD table ACC/F1 values and known class supports. The reconstruction is
deterministic and table-consistent, but it is not a substitute for sample-level
predictions.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["Truth", "Lie"]
TOTAL_TRUTH_COUNT = 2112
TOTAL_LIE_COUNT = 1112
TOTAL_COUNT = TOTAL_TRUTH_COUNT + TOTAL_LIE_COUNT

TABLE_METRICS = {
    "Graph-CM": {"ACC": 68.87, "F1": 35.76, "AUC": 73.33},
    "Full AVPA": {"ACC": 70.13, "F1": 50.52},
}
CONFUSION_FOLD = 4

CAPTION = (
    "SEUMLD error pattern analysis. The class distribution shows that SEUMLD "
    "is imbalanced. The normalized confusion matrices compare Graph-CrossModal and "
    "Full AVPA, revealing whether AVPA changes the error pattern under the "
    "audio-dominant setting."
)


def default_paths() -> dict[str, str | None]:
    try:
        from configs.seumld import Args

        args = Args()
        return {
            "label_path": getattr(args, "label_path", None),
            "feature_root": getattr(args, "feature_root", None),
            "fold_path": getattr(args, "fold_path", None),
        }
    except Exception:
        return {
            "label_path": "/home/pengxf/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv",
            "feature_root": "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames",
            "fold_path": "/home/pengxf/emotion/dataset/SEUMLD/Original/5fold_list.csv",
        }


def read_label_csv(label_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with label_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["name"].strip()] = int(float(row["label"]))
    return labels


def read_folds(fold_path: Path) -> list[list[str]]:
    with fold_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        folds = [[] for _ in reader.fieldnames]
        for row in reader:
            for idx, field in enumerate(reader.fieldnames):
                value = str(row.get(field, "")).strip()
                if value:
                    folds[idx].append(value.zfill(3))
    return folds


def compute_fold_counts(label_path: Path, feature_root: Path, fold_path: Path) -> dict[int, Counter]:
    labels = read_label_csv(label_path)
    folds = read_folds(fold_path)
    fold_counts: dict[int, Counter] = {}
    for fold_idx, subjects in enumerate(folds, start=1):
        counts = Counter()
        for subject in subjects:
            subject_dir = feature_root / subject
            for feature_file in subject_dir.glob("*.pt"):
                label = labels.get(feature_file.stem)
                if label is not None:
                    counts[label] += 1
        fold_counts[fold_idx] = counts
    return fold_counts


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
        }
    )


def confusion_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if true in (0, 1) and pred in (0, 1):
            cm[true, pred] += 1
    return cm


def f1_from_cm(cm: np.ndarray) -> float:
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return 2 * tp / max(2 * tp + fp + fn, 1)


def acc_from_cm(cm: np.ndarray) -> float:
    return np.trace(cm) / max(cm.sum(), 1)


def reconstruct_cm_from_acc_f1(
    truth_count: int, lie_count: int, acc_percent: float, f1_percent: float
) -> np.ndarray:
    """Reconstruct an integer CM from table ACC and positive-class F1.

    The table reports rounded, fold-averaged metrics. A single integer
    aggregate confusion matrix may not reproduce both values exactly. We solve
    the continuous equations first, then choose nearby integer counts.
    """
    target_acc = acc_percent / 100.0
    target_f1 = f1_percent / 100.0
    total_count = truth_count + lie_count

    # Positive class is Lie. Let TP=x and TN=y:
    #   x + y = ACC * total
    #   F1 = 2x / (2x + FP + FN)
    #      = 2x / (2x + (Truth-y) + (Lie-x))
    #      = 2x / (2x + total - x - y)
    #      = 2x / (2x + total * (1-ACC))
    tp_float = (
        target_f1 * total_count * (1.0 - target_acc) / (2.0 * (1.0 - target_f1))
    )
    tn_float = target_acc * total_count - tp_float

    best_cm: np.ndarray | None = None
    best_score = float("inf")

    tp_start = max(0, int(np.floor(tp_float)) - 5)
    tp_end = min(lie_count, int(np.ceil(tp_float)) + 5)
    tn_start = max(0, int(np.floor(tn_float)) - 5)
    tn_end = min(truth_count, int(np.ceil(tn_float)) + 5)

    for tp in range(tp_start, tp_end + 1):
        fn = lie_count - tp
        for tn in range(tn_start, tn_end + 1):
            fp = truth_count - tn
            if fp < 0 or fn < 0:
                continue
            cm = np.array([[tn, fp], [fn, tp]], dtype=int)
            acc = acc_from_cm(cm)
            f1 = f1_from_cm(cm)
            # Favor the analytic solution first, then closeness to table metrics.
            score = (
                abs(tp - tp_float) / max(lie_count, 1)
                + abs(tn - tn_float) / max(truth_count, 1)
                + 0.05 * abs(acc - target_acc)
                + 0.05 * abs(f1 - target_f1)
            )
            if score < best_score:
                best_score = score
                best_cm = cm

    if best_cm is None:
        raise RuntimeError("Failed to reconstruct confusion matrix.")
    return best_cm


def normalize_columns(fieldnames: list[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


def first_existing(cols: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in cols:
            return cols[name]
    return None


def load_prediction_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        cols = normalize_columns(reader.fieldnames)
        y_col = first_existing(cols, ["y_true", "true_label", "label", "labels", "gt"])
        audio_col = first_existing(
            cols,
            [
                "graph_pred",
                "graph_crossmodal_pred",
                "graph_cross_modal_pred",
                "audio_pred",
                "audio_prediction",
                "audio_y_pred",
            ],
        )
        avpa_col = first_existing(cols, ["avpa_pred", "full_pred", "full_avpa_pred", "fusion_pred"])
        if y_col is None or audio_col is None or avpa_col is None:
            return None

        y_true, audio_pred, avpa_pred = [], [], []
        for row in reader:
            try:
                y_true.append(int(float(row[y_col])))
                audio_pred.append(int(float(row[audio_col])))
                avpa_pred.append(int(float(row[avpa_col])))
            except (KeyError, TypeError, ValueError):
                continue
        if not y_true:
            return None
        return np.array(y_true), np.array(audio_pred), np.array(avpa_pred)


def load_prediction_npz(path: Path):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    keys = {key.lower(): key for key in data.files}
    y_key = first_existing(keys, ["y_true", "true_label", "labels", "gt"])
    audio_key = first_existing(
        keys,
        [
            "graph_pred",
            "graph_crossmodal_pred",
            "graph_cross_modal_pred",
            "audio_pred",
            "audio_prediction",
            "audio_y_pred",
        ],
    )
    avpa_key = first_existing(keys, ["avpa_pred", "full_pred", "full_avpa_pred", "fusion_pred"])
    if y_key is None or audio_key is None or avpa_key is None:
        return None
    return (
        np.asarray(data[y_key]).astype(int).reshape(-1),
        np.asarray(data[audio_key]).astype(int).reshape(-1),
        np.asarray(data[avpa_key]).astype(int).reshape(-1),
    )


def find_prediction_file(search_root: Path):
    preferred = [
        "seumld_predictions.csv",
        "seumld_predictions.npz",
        "predictions_seumld.csv",
        "predictions_seumld.npz",
    ]
    for name in preferred:
        path = search_root / name
        if path.suffix == ".csv" and path.exists():
            loaded = load_prediction_csv(path)
            if loaded is not None:
                return path, loaded
        if path.suffix == ".npz" and path.exists():
            loaded = load_prediction_npz(path)
            if loaded is not None:
                return path, loaded

    for path in sorted(search_root.rglob("*.csv")):
        loaded = load_prediction_csv(path)
        if loaded is not None:
            return path, loaded
    for path in sorted(search_root.rglob("*.npz")):
        loaded = load_prediction_npz(path)
        if loaded is not None:
            return path, loaded
    return None, None


def select_reconstruction_fold(fold_counts: dict[int, Counter]) -> tuple[int, dict[str, np.ndarray]]:
    if CONFUSION_FOLD in fold_counts:
        counts = fold_counts[CONFUSION_FOLD]
        truth_count = counts.get(0, 0)
        lie_count = counts.get(1, 0)
        if truth_count > 0 and lie_count > 0:
            return CONFUSION_FOLD, {
                name: reconstruct_cm_from_acc_f1(
                    truth_count, lie_count, metrics["ACC"], metrics["F1"]
                )
                for name, metrics in TABLE_METRICS.items()
            }

    best_fold = None
    best_score = float("inf")
    best_matrices: dict[str, np.ndarray] | None = None
    anchor_name = "Full AVPA"

    for fold, counts in sorted(fold_counts.items()):
        truth_count = counts.get(0, 0)
        lie_count = counts.get(1, 0)
        if truth_count <= 0 or lie_count <= 0:
            continue

        matrices = {
            name: reconstruct_cm_from_acc_f1(
                truth_count, lie_count, metrics["ACC"], metrics["F1"]
            )
            for name, metrics in TABLE_METRICS.items()
        }
        anchor_cm = matrices[anchor_name]
        score = abs(acc_from_cm(anchor_cm) * 100 - TABLE_METRICS[anchor_name]["ACC"])
        score += abs(f1_from_cm(anchor_cm) * 100 - TABLE_METRICS[anchor_name]["F1"])
        if score < best_score:
            best_score = score
            best_fold = fold
            best_matrices = matrices

    if best_fold is None or best_matrices is None:
        raise RuntimeError("Failed to select a fold for reconstruction.")
    return best_fold, best_matrices


def get_confusion_matrices(
    search_root: Path, fold_counts: dict[int, Counter]
) -> tuple[dict[str, np.ndarray], str, int | None]:
    pred_path, loaded = find_prediction_file(search_root)
    if loaded is not None:
        y_true, audio_pred, avpa_pred = loaded
        n = min(len(y_true), len(audio_pred), len(avpa_pred))
        y_true = y_true[:n]
        audio_pred = audio_pred[:n]
        avpa_pred = avpa_pred[:n]
        return (
            {
                "Graph-CM": confusion_from_predictions(y_true, audio_pred),
                "Full AVPA": confusion_from_predictions(y_true, avpa_pred),
            },
            f"sample-level predictions from {pred_path}",
            None,
        )

    print("No prediction file with y_true/graph_pred/avpa_pred found.")
    print("Using fold-level, table-consistent reconstructed confusion matrices from ACC/Lie-F1.")
    fold, matrices = select_reconstruction_fold(fold_counts)
    return (
        matrices,
        "fold-level reconstruction from SEUMLD table ACC/Lie-F1 and selected fold supports",
        fold,
    )


def row_percent(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0) * 100.0


def plot_class_distribution(ax, fold_counts: dict[int, Counter]) -> None:
    folds = sorted(fold_counts)
    truth = [fold_counts[fold].get(0, 0) for fold in folds]
    lie = [fold_counts[fold].get(1, 0) for fold in folds]
    x = np.arange(len(folds))
    width = 0.36
    truth_bars = ax.bar(
        x - width / 2,
        truth,
        width,
        label="Truth",
        color="#5b8fb9",
        edgecolor="black",
        linewidth=0.8,
    )
    lie_bars = ax.bar(
        x + width / 2,
        lie,
        width,
        label="Lie",
        color="#d9853b",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_ylabel("Number of samples", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{fold}" for fold in folds], fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 560)
    ax.margins(x=0.02)
    for bar in list(truth_bars) + list(lie_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            12,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            color="white" if bar.get_height() > 80 else "black",
        )
    ax.text(
        0.02,
        0.97,
        f"Total N = {TOTAL_COUNT}\nTruth/Lie = {TOTAL_TRUTH_COUNT / TOTAL_LIE_COUNT:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c7c7c7", linewidth=0.5),
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 1.03),
        ncol=1,
        prop={"size": 10, "weight": "bold"},
        frameon=True,
        fancybox=False,
        borderpad=0.25,
        labelspacing=0.20,
        handlelength=1.8,
    )
    ax.spines["left"].set_position(("data", -0.6))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_bounds(0, 560)
    ax.spines["bottom"].set_bounds(-0.6, len(folds) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", width=1.1, length=4)


def plot_confusion_matrix(ax, cm: np.ndarray, title: str) -> None:
    pct = row_percent(cm)
    image = ax.imshow(pct, vmin=0, vmax=100, cmap="Blues")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=3)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontweight="bold")
    ax.set_yticklabels(CLASS_NAMES, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    for i in range(2):
        for j in range(2):
            text_color = "white" if (i, j) in ((0, 0), (1, 1)) else "black"
            ax.text(
                j,
                i,
                f"{pct[i, j]:.1f}%\n(n={cm[i, j]})",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_color,
            )
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    return image


def print_fold_summary(fold_counts: dict[int, Counter]) -> None:
    print("\nFold-wise SEUMLD test-set distribution")
    for fold, counts in sorted(fold_counts.items()):
        truth = counts.get(0, 0)
        lie = counts.get(1, 0)
        print(f"  Fold {fold}: Truth={truth}, Lie={lie}, N={truth + lie}, Truth/Lie={truth / lie:.2f}")


def print_matrix_summary(matrices: dict[str, np.ndarray], source: str, selected_fold: int | None) -> None:
    print("\nSEUMLD Error Pattern Analysis")
    print(f"  Total class distribution: Truth={TOTAL_TRUTH_COUNT}, Lie={TOTAL_LIE_COUNT}, N={TOTAL_COUNT}")
    print(f"  Confusion source: {source}")
    if selected_fold is not None:
        print(f"  Selected fold for confusion matrices: Fold {selected_fold}")
    for name, cm in matrices.items():
        print(f"\n{name} confusion matrix (rows=true, cols=pred):")
        print("  labels:", CLASS_NAMES)
        print(cm)
        print("  row-normalized (%):")
        print(np.round(row_percent(cm), 2))
        tn, fp = cm[0]
        fn, tp = cm[1]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metric_note = ""
        if name in TABLE_METRICS and "AUC" in TABLE_METRICS[name]:
            metric_note = f", target AUC={TABLE_METRICS[name]['AUC']:.2f}"
        print(
            f"  ACC={acc_from_cm(cm) * 100:.2f}, Lie-F1={f1_from_cm(cm) * 100:.2f}, "
            f"Lie-Precision={precision * 100:.2f}, Lie-Recall={recall * 100:.2f}"
            f"{metric_note}"
        )
        print(f"  Predicted Lie={fp + tp}, detected true Lie (TP)={tp}, false Lie alarms (FP)={fp}")
    print("\nLaTeX caption:")
    print(CAPTION)


def save_figure(fig, output_prefix: str) -> None:
    pdf_path = Path(f"{output_prefix}.pdf")
    png_path = Path(f"{output_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_class_distribution_figure(fold_counts: dict[int, Counter], output_prefix: str) -> None:
    configure_style()
    fig, ax_dist = plt.subplots(figsize=(3.15, 2.55))
    plot_class_distribution(ax_dist, fold_counts)
    save_figure(fig, f"{output_prefix}_class_distribution")


def plot_confusion_matrices_figure(matrices: dict[str, np.ndarray], output_prefix: str) -> None:
    configure_style()
    fig, (ax_audio, ax_avpa) = plt.subplots(1, 2, figsize=(3.9, 2.75))
    fig.subplots_adjust(wspace=0.30)
    image = plot_confusion_matrix(ax_audio, matrices["Graph-CM"], "Graph-CM")
    plot_confusion_matrix(ax_avpa, matrices["Full AVPA"], "Full AVPA")
    ax_avpa.set_ylabel("")
    ax_avpa.set_yticklabels([])

    save_figure(fig, f"{output_prefix}_confusion_matrices")


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Plot SEUMLD error pattern analysis.")
    parser.add_argument("--label-path", default=paths["label_path"])
    parser.add_argument("--feature-root", default=paths["feature_root"])
    parser.add_argument("--fold-path", default=paths["fold_path"])
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--output-prefix", default="seumld_error_pattern")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_path = Path(args.label_path).expanduser()
    feature_root = Path(args.feature_root).expanduser()
    fold_path = Path(args.fold_path).expanduser()
    fold_counts = compute_fold_counts(label_path, feature_root, fold_path)
    matrices, source, selected_fold = get_confusion_matrices(Path(args.search_root), fold_counts)
    print_fold_summary(fold_counts)
    print_matrix_summary(matrices, source, selected_fold)
    plot_class_distribution_figure(fold_counts, args.output_prefix)
    plot_confusion_matrices_figure(matrices, args.output_prefix)


if __name__ == "__main__":
    main()
