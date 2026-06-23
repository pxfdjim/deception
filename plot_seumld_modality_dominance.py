#!/usr/bin/env python3
"""
Create the rebuttal figure:
SEUMLD Modality-Dominance Analysis.

Outputs:
  seumld_modality_dominance.pdf
  seumld_modality_dominance.png
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PERFORMANCE = {
    "Visual Only": {"ACC": 65.57, "F1": 9.19, "AUC": 49.74},
    "Audio Only": {"ACC": 64.80, "F1": 49.60, "AUC": 69.00},
    "Full AVPA": {"ACC": 69.65, "F1": 49.78, "AUC": 69.20},
}

ABLATION_DETAILS_CANDIDATES = [
    Path("experiments/hierarchical_mean/ablation_20260319_103618/ablation_fold_details.csv"),
]

LATEX_CAPTION = (
    "SEUMLD Modality-Dominance Analysis. The SEUMLD benchmark exhibits an "
    "audio-dominant pattern: visual-only performance is weak, while Audio Only "
    "already approaches Full AVPA in class-wise F1/Recall. This suggests that discriminative "
    "cues in SEUMLD are more strongly reflected in acoustic-prosodic patterns "
    "than in subtle visual behaviors."
)


def default_paths() -> dict[str, str | None]:
    paths = {
        "label_path": None,
        "feature_root": None,
        "fold_path": None,
    }
    try:
        from configs.seumld import Args

        args = Args()
        paths["label_path"] = getattr(args, "label_path", None)
        paths["feature_root"] = getattr(args, "feature_root", None)
        paths["fold_path"] = getattr(args, "fold_path", None)
    except Exception:
        paths["label_path"] = (
            "/home/pengxf/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv"
        )
        paths["feature_root"] = (
            "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames"
        )
        paths["fold_path"] = (
            "/home/pengxf/emotion/dataset/SEUMLD/Original/5fold_list.csv"
        )
    return paths


def infer_label_names() -> tuple[dict[int, str], bool]:
    dataset_file = Path(__file__).resolve().parent / "datasets" / "seumld.py"
    if dataset_file.exists():
        text = dataset_file.read_text(encoding="utf-8", errors="ignore").lower()
        zero_is_truth = re.search(r"0\s*=\s*[^\n]*(truth|真实)", text) is not None
        one_is_lie = re.search(r"1\s*=\s*[^\n]*(lie|谎言)", text) is not None
        zero_is_lie = re.search(r"0\s*=\s*[^\n]*(lie|谎言)", text) is not None
        one_is_truth = re.search(r"1\s*=\s*[^\n]*(truth|真实)", text) is not None
        if zero_is_truth and one_is_lie:
            return {0: "Truth", 1: "Lie"}, True
        if zero_is_lie and one_is_truth:
            return {0: "Lie", 1: "Truth"}, True
    return {0: "Class 0", 1: "Class 1"}, False


def read_label_csv(label_path: Path) -> dict[str, int]:
    if not label_path.exists():
        raise FileNotFoundError(
            f"SEUMLD label file not found: {label_path}. "
            "Please specify it with --label-path."
        )

    label_map: dict[str, int] = {}
    with label_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {label_path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "name" not in fields or "label" not in fields:
            raise ValueError(
                f"Expected columns 'name' and 'label' in {label_path}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            name = row[fields["name"]].strip()
            label_text = row[fields["label"]].strip()
            if not name or label_text == "":
                continue
            label_map[name] = int(float(label_text))
    return label_map


def read_fold_subjects(fold_path: Path) -> set[str]:
    if not fold_path.exists():
        return set()
    subjects: set[str] = set()
    with fold_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for value in row.values():
                value = str(value).strip()
                if value:
                    subjects.add(value.zfill(3))
    return subjects


def read_folds(fold_path: Path) -> list[list[str]]:
    if not fold_path.exists():
        return []
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


def collect_feature_video_names(feature_root: Path, fold_path: Path) -> set[str]:
    if not feature_root.exists():
        return set()

    fold_subjects = read_fold_subjects(fold_path)
    names: set[str] = set()
    for feature_file in feature_root.glob("*/*.pt"):
        if fold_subjects and feature_file.parent.name not in fold_subjects:
            continue
        names.add(feature_file.stem)
    return names


def compute_class_distribution(
    label_path: Path, feature_root: Path | None, fold_path: Path | None
) -> tuple[Counter, int, str]:
    label_map = read_label_csv(label_path)
    source = f"label CSV ({label_path})"

    feature_names: set[str] = set()
    if feature_root is not None and fold_path is not None:
        feature_names = collect_feature_video_names(feature_root, fold_path)

    if feature_names:
        labels = [label_map[name] for name in sorted(feature_names) if name in label_map]
        missing = len(feature_names) - len(labels)
        source = (
            f"label CSV intersected with feature files and fold list "
            f"({len(feature_names)} feature files, {missing} without label)"
        )
    else:
        labels = list(label_map.values())

    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("No SEUMLD labels were found after filtering.")
    return counts, total, source


def compute_fold_class_counts(
    label_path: Path, feature_root: Path | None, fold_path: Path | None
) -> dict[int, Counter]:
    if feature_root is None or fold_path is None or not feature_root.exists() or not fold_path.exists():
        return {}

    label_map = read_label_csv(label_path)
    folds = read_folds(fold_path)
    fold_counts: dict[int, Counter] = {}
    for fold_idx, subjects in enumerate(folds, start=1):
        subject_set = set(subjects)
        counts = Counter()
        for feature_file in feature_root.glob("*/*.pt"):
            if feature_file.parent.name not in subject_set:
                continue
            label = label_map.get(feature_file.stem)
            if label is not None:
                counts[label] += 1
        fold_counts[fold_idx] = counts
    return fold_counts


def find_ablation_details(search_root: Path) -> Path | None:
    for rel_path in ABLATION_DETAILS_CANDIDATES:
        path = search_root / rel_path
        if path.exists():
            return path
    matches = sorted(search_root.rglob("ablation_fold_details.csv"))
    for path in matches:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "only_audio" in text and "full" in text:
            return path
    return None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def load_panel_b_per_class_metrics(
    search_root: Path, fold_counts: dict[int, Counter], label_names: dict[int, str]
) -> tuple[dict[str, dict[str, dict[str, float]]] | None, Path | None]:
    details_path = find_ablation_details(search_root)
    if details_path is None or not fold_counts:
        return None, details_path

    target_modes = {"only_audio": "Audio Only", "full": "Full AVPA"}
    accum: dict[str, dict[str, list[float]]] = {
        "Audio Only": {"Truth_F1": [], "Truth_Recall": [], "Lie_F1": [], "Lie_Recall": []},
        "Full AVPA": {"Truth_F1": [], "Truth_Recall": [], "Lie_F1": [], "Lie_Recall": []},
    }

    with details_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode", "").strip()
            if mode not in target_modes:
                continue
            try:
                fold = int(float(row["fold"]))
                lie_precision = float(row["precision"])
                lie_recall = float(row["recall"])
                lie_f1 = float(row["f1"])
            except (KeyError, TypeError, ValueError):
                continue

            supports = fold_counts.get(fold, Counter())
            n_truth = supports.get(0, 0)
            n_lie = supports.get(1, 0)
            if n_truth <= 0 or n_lie <= 0:
                continue

            tp = lie_recall * n_lie
            fn = n_lie - tp
            fp = safe_div(tp, lie_precision) - tp if lie_precision > 0 else 0.0
            fp = max(fp, 0.0)
            tn = max(n_truth - fp, 0.0)

            truth_recall = safe_div(tn, n_truth)
            truth_precision = safe_div(tn, tn + fn)
            truth_f1 = f1_from_precision_recall(truth_precision, truth_recall)

            model_name = target_modes[mode]
            accum[model_name]["Truth_F1"].append(truth_f1 * 100.0)
            accum[model_name]["Truth_Recall"].append(truth_recall * 100.0)
            accum[model_name]["Lie_F1"].append(lie_f1 * 100.0)
            accum[model_name]["Lie_Recall"].append(lie_recall * 100.0)

    if not accum["Audio Only"]["Lie_F1"] or not accum["Full AVPA"]["Lie_F1"]:
        return None, details_path

    panel_metrics = {}
    for model_name, values in accum.items():
        panel_metrics[model_name] = {
            "Truth": {
                "F1": float(np.mean(values["Truth_F1"])),
                "Recall": float(np.mean(values["Truth_Recall"])),
            },
            "Lie": {
                "F1": float(np.mean(values["Lie_F1"])),
                "Recall": float(np.mean(values["Lie_Recall"])),
            },
        }

    print(f"\nPanel (b) per-class metrics source: {details_path}")
    print("  Truth metrics are reconstructed from fold-level Lie precision/recall and fold supports.")
    print("  No sample-level confusion matrix is drawn.")
    for model_name, classes in panel_metrics.items():
        print(f"  {model_name}:")
        for class_name in [label_names.get(0, "Truth"), label_names.get(1, "Lie")]:
            # Panel labels intentionally use Truth/Lie after mapping inference.
            key = "Truth" if class_name == label_names.get(0, "Truth") else "Lie"
            print(
                f"    {class_name}: F1={classes[key]['F1']:.2f}, "
                f"Recall={classes[key]['Recall']:.2f}"
            )
    return panel_metrics, details_path


def ratio_text(counts: Counter, label_names: dict[int, str]) -> str:
    labels = sorted(counts)
    if len(labels) == 2 and counts[labels[1]] > 0:
        return (
            f"{label_names.get(labels[0], labels[0])}/{label_names.get(labels[1], labels[1])} "
            f"= {counts[labels[0]] / counts[labels[1]]:.2f}"
        )
    return ", ".join(f"{label_names.get(k, k)}={v}" for k, v in sorted(counts.items()))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def annotate_bars(ax, bars, fmt="{:.2f}", dy=1.2, fontsize=6.5) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_figure(
    counts: Counter,
    total: int,
    label_names: dict[int, str],
    output_prefix: str,
    panel_b_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    configure_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fig.suptitle("SEUMLD Modality-Dominance Analysis", fontsize=10.5, fontweight="bold", y=1.02)

    ax = axes[0]
    class_ids = sorted(counts)
    class_labels = [label_names.get(i, f"Class {i}") for i in class_ids]
    class_counts = [counts[i] for i in class_ids]
    colors = ["#5b8fb9", "#d9853b", "#8a8a8a", "#7aa974"][: len(class_ids)]
    bars = ax.bar(class_labels, class_counts, color=colors, width=0.58, edgecolor="black", linewidth=0.6)
    annotate_bars(ax, bars, fmt="{:.0f}", dy=max(class_counts) * 0.015, fontsize=7)
    ax.set_title("(a) Class distribution", fontweight="bold", pad=5)
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)
    ax.text(
        0.97,
        0.93,
        f"N = {total}\n{ratio_text(counts, label_names)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c7c7c7", linewidth=0.5),
    )

    ax = axes[1]
    if panel_b_metrics is not None:
        groups = [
            ("Truth\nF1", "Truth", "F1"),
            ("Truth\nRecall", "Truth", "Recall"),
            ("Lie\nF1", "Lie", "F1"),
            ("Lie\nRecall", "Lie", "Recall"),
        ]
        model_names = ["Audio Only", "Full AVPA"]
        x = np.arange(len(groups))
        width = 0.34
        audio_vals = [panel_b_metrics["Audio Only"][cls][metric] for _, cls, metric in groups]
        avpa_vals = [panel_b_metrics["Full AVPA"][cls][metric] for _, cls, metric in groups]
        audio_bars = ax.bar(
            x - width / 2,
            audio_vals,
            width,
            label=model_names[0],
            color="#4f79a7",
            edgecolor="black",
            linewidth=0.5,
        )
        avpa_bars = ax.bar(
            x + width / 2,
            avpa_vals,
            width,
            label=model_names[1],
            color="#e0a143",
            edgecolor="black",
            linewidth=0.5,
        )
        annotate_bars(ax, audio_bars, dy=1.0, fontsize=5.8)
        annotate_bars(ax, avpa_bars, dy=1.0, fontsize=5.8)
        ax.set_title("(b) Audio Only vs AVPA per-class metrics", fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in groups])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", frameon=True, fancybox=False, borderpad=0.3)
    else:
        model_names = list(PERFORMANCE.keys())
        x = np.arange(len(model_names))
        width = 0.32
        f1_vals = [PERFORMANCE[name]["F1"] for name in model_names]
        auc_vals = [PERFORMANCE[name]["AUC"] for name in model_names]
        f1_bars = ax.bar(
            x - width / 2,
            f1_vals,
            width,
            label="F1",
            color="#4f79a7",
            edgecolor="black",
            linewidth=0.5,
        )
        auc_bars = ax.bar(
            x + width / 2,
            auc_vals,
            width,
            label="AUC",
            color="#e0a143",
            edgecolor="black",
            linewidth=0.5,
        )
        annotate_bars(ax, f1_bars)
        annotate_bars(ax, auc_bars)
        ax.set_title("(b) Modality performance comparison", fontweight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Visual\nOnly", "Audio\nOnly", "Full\nAVPA"])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=True, fancybox=False, borderpad=0.3)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.5)
    pdf_path = Path(f"{output_prefix}.pdf")
    png_path = Path(f"{output_prefix}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def normalize_columns(fieldnames: list[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


def first_existing(cols: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in cols:
            return cols[name]
    return None


def labels_from_probability(values: list[float]) -> list[int]:
    return [1 if value >= 0.5 else 0 for value in values]


def try_load_prediction_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        cols = normalize_columns(reader.fieldnames)
        y_col = first_existing(cols, ["y_true", "true_label", "label", "labels", "gt"])
        audio_pred_col = first_existing(cols, ["audio_pred", "audio_prediction", "audio_y_pred"])
        audio_prob_col = first_existing(cols, ["audio_prob", "audio_score", "audio_p_lie", "audio_prob_lie"])
        avpa_pred_col = first_existing(
            cols, ["avpa_pred", "full_pred", "full_avpa_pred", "fusion_pred", "pred"]
        )
        avpa_prob_col = first_existing(
            cols, ["avpa_prob", "full_prob", "full_avpa_prob", "fusion_prob", "prob"]
        )
        if y_col is None or (audio_pred_col is None and audio_prob_col is None):
            return None
        if avpa_pred_col is None and avpa_prob_col is None:
            return None

        y_true: list[int] = []
        audio_values: list[float] = []
        avpa_values: list[float] = []
        for row in reader:
            try:
                y_true.append(int(float(row[y_col])))
                if audio_pred_col is not None:
                    audio_values.append(int(float(row[audio_pred_col])))
                else:
                    audio_values.append(float(row[audio_prob_col]))
                if avpa_pred_col is not None:
                    avpa_values.append(int(float(row[avpa_pred_col])))
                else:
                    avpa_values.append(float(row[avpa_prob_col]))
            except (KeyError, TypeError, ValueError):
                continue

    if not y_true:
        return None
    audio_pred = (
        [int(v) for v in audio_values]
        if audio_pred_col is not None
        else labels_from_probability(audio_values)
    )
    avpa_pred = (
        [int(v) for v in avpa_values]
        if avpa_pred_col is not None
        else labels_from_probability(avpa_values)
    )
    return y_true, audio_pred, avpa_pred


def try_load_prediction_npz(path: Path):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    keys = {key.lower(): key for key in data.files}
    y_key = first_existing(keys, ["y_true", "true_label", "labels", "gt"])
    audio_pred_key = first_existing(keys, ["audio_pred", "audio_prediction", "audio_y_pred"])
    audio_prob_key = first_existing(keys, ["audio_prob", "audio_score", "audio_p_lie", "audio_prob_lie"])
    avpa_pred_key = first_existing(keys, ["avpa_pred", "full_pred", "full_avpa_pred", "fusion_pred"])
    avpa_prob_key = first_existing(keys, ["avpa_prob", "full_prob", "full_avpa_prob", "fusion_prob"])
    if y_key is None or (audio_pred_key is None and audio_prob_key is None):
        return None
    if avpa_pred_key is None and avpa_prob_key is None:
        return None

    y_true = np.asarray(data[y_key]).astype(int).reshape(-1).tolist()
    if audio_pred_key is not None:
        audio_pred = np.asarray(data[audio_pred_key]).astype(int).reshape(-1).tolist()
    else:
        audio_pred = labels_from_probability(np.asarray(data[audio_prob_key]).reshape(-1).tolist())
    if avpa_pred_key is not None:
        avpa_pred = np.asarray(data[avpa_pred_key]).astype(int).reshape(-1).tolist()
    else:
        avpa_pred = labels_from_probability(np.asarray(data[avpa_prob_key]).reshape(-1).tolist())
    return y_true, audio_pred, avpa_pred


def find_prediction_data(search_root: Path):
    csv_files = sorted(search_root.rglob("*.csv"))
    npz_files = sorted(search_root.rglob("*.npz"))
    for path in csv_files:
        loaded = try_load_prediction_csv(path)
        if loaded is not None:
            return path, loaded
    for path in npz_files:
        loaded = try_load_prediction_npz(path)
        if loaded is not None:
            return path, loaded
    return None, None


def confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[int]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    return cm


def print_per_class_metrics(
    title: str, y_true: list[int], y_pred: list[int], label_names: dict[int, str]
) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels)
    print(f"\n{title} per-class Precision / Recall / F1:")
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(
            f"  {label_names.get(label, f'Class {label}')}: "
            f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )
    print(f"{title} confusion matrix (rows=true, cols=pred):")
    print("  labels:", [label_names.get(label, f"Class {label}") for label in labels])
    print(cm)


def maybe_print_prediction_analysis(search_root: Path, label_names: dict[int, str]) -> None:
    path, data = find_prediction_data(search_root)
    if data is None:
        print("\nNo prediction file found. Per-class and confusion analysis skipped.")
        return
    y_true, audio_pred, avpa_pred = data
    n = min(len(y_true), len(audio_pred), len(avpa_pred))
    y_true = y_true[:n]
    audio_pred = audio_pred[:n]
    avpa_pred = avpa_pred[:n]
    print(f"\nPrediction file found: {path}")
    print_per_class_metrics("Audio Only", y_true, audio_pred, label_names)
    print_per_class_metrics("Full AVPA", y_true, avpa_pred, label_names)


def print_summary(
    counts: Counter,
    total: int,
    source: str,
    label_names: dict[int, str],
    mapping_known: bool,
) -> None:
    print("\nSEUMLD class distribution")
    print(f"  Source: {source}")
    print(f"  Total samples: {total}")
    for label in sorted(counts):
        name = label_names.get(label, f"Class {label}")
        pct = counts[label] / total * 100.0
        print(f"  {name} ({label}): {counts[label]} ({pct:.2f}%)")
    print(f"  Class ratio: {ratio_text(counts, label_names)}")
    if not mapping_known:
        print("Please verify the label mapping before using Truth/Lie names.")

    print("\nModality performance values on SEUMLD")
    for model_name, metrics in PERFORMANCE.items():
        print(
            f"  {model_name}: "
            f"ACC={metrics['ACC']:.2f}, F1={metrics['F1']:.2f}, AUC={metrics['AUC']:.2f}"
        )

    print("\nLaTeX caption:")
    print(LATEX_CAPTION)


def parse_args():
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description="Plot SEUMLD modality-dominance rebuttal figure."
    )
    parser.add_argument("--label-path", default=paths["label_path"])
    parser.add_argument("--feature-root", default=paths["feature_root"])
    parser.add_argument("--fold-path", default=paths["fold_path"])
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--output-prefix", default="seumld_modality_dominance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_path = Path(args.label_path).expanduser()
    feature_root = Path(args.feature_root).expanduser() if args.feature_root else None
    fold_path = Path(args.fold_path).expanduser() if args.fold_path else None

    label_names, mapping_known = infer_label_names()
    counts, total, source = compute_class_distribution(label_path, feature_root, fold_path)
    fold_counts = compute_fold_class_counts(label_path, feature_root, fold_path)
    panel_b_metrics, _ = load_panel_b_per_class_metrics(
        Path(args.search_root), fold_counts, label_names
    )
    print_summary(counts, total, source, label_names, mapping_known)
    maybe_print_prediction_analysis(Path(args.search_root), label_names)
    plot_figure(counts, total, label_names, args.output_prefix, panel_b_metrics)


if __name__ == "__main__":
    main()
