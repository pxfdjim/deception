#!/usr/bin/env python3
"""
Leave-one-dataset-out evaluation for DOLOS, SEUMLD, and Real-Life.

Each experiment trains on two complete source datasets and evaluates once on
the complete held-out target dataset. The held-out dataset is never used for
training, validation, checkpoint selection, or early stopping.

Examples:
    python cross_dataset_test.py --target reallife
    python cross_dataset_test.py --target dolos --epochs 50
    python cross_dataset_test.py --target all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from configs.dolos import Args as DOLOSArgs
from configs.real_life import Args as RealLifeArgs
from configs.seumld import Args as SEUMLDArgs
from datasets.dolos import DOLOSDataset
from datasets.real_life import RealLifeLOOCVDataset
from datasets.seumld import SEUMLDFineGrainedDataset
from models.final import LieDetection
from utils.utils import set_random_seed


DATASET_NAMES = ("dolos", "seumld", "reallife")


@dataclass
class ExperimentOptions:
    """Shared cross-dataset settings for the common audiovisual model."""

    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 1
    lr: float = 1e-3
    weight_decay: float = 5e-6
    momentum: float = 0.9
    seed: int = 42
    hidden_dim: int = 128
    low_dim: int = 64
    proto_m: float = 0.9
    ortho_weight: float = 0.1
    output_dir: str = "exper_model/cross_dataset"
    device: str = "cuda"
    dry_run: bool = False

    # LieDetection model interface.
    modality: str = "both"
    visual_dim: int = 768
    audio_dim: int = 1024
    num_classes: int = 2
    use_visual_self_attn: bool = False
    visual_self_attn_heads: int = 4
    visual_self_attn_layers: int = 1
    visual_self_attn_dropout: float = 0.1
    use_instance_loss: bool = True
    instance_loss_weight: float = 0.1
    positive_instance_topk_ratio: float = 0.25


class DomainDataset(Dataset):
    """Adds a domain name while preserving each existing dataset loader."""

    def __init__(self, domain: str, dataset: Dataset):
        self.domain = domain
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["domain"] = self.domain
        return item


class IndexedDataset(Dataset):
    """Indexes a deduplicated union of existing dataset partitions."""

    def __init__(self, references: Sequence[tuple[Dataset, int]]):
        self.references = list(references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> Any:
        dataset, local_index = self.references[index]
        return dataset[local_index]


def _label_value(label: Any) -> int:
    return int(label.item()) if torch.is_tensor(label) else int(label)


def _sample_id(item: dict[str, Any]) -> str:
    if "video_name" in item:
        return str(item["video_name"])
    if "id" in item:
        return str(item["id"])
    meta = item.get("meta", {})
    return f"{meta.get('subj', 'unknown')}_q{meta.get('qid', 'unknown')}"


def _normalize_visual_feature(feature: Any, visual_dim: int) -> torch.Tensor:
    if feature is None:
        return torch.zeros(1, visual_dim)
    tensor = feature.float() if torch.is_tensor(feature) else torch.as_tensor(feature, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != visual_dim:
        raise ValueError(f"Expected visual feature [segments, {visual_dim}], got {tuple(tensor.shape)}")
    return tensor


def _normalize_audio_feature(feature: Any, audio_dim: int) -> torch.Tensor:
    if feature is None:
        return torch.zeros(1, audio_dim)
    tensor = feature.float() if torch.is_tensor(feature) else torch.as_tensor(feature, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim != 2 or tensor.shape[-1] != audio_dim:
        raise ValueError(f"Expected audio feature [frames, {audio_dim}], got {tuple(tensor.shape)}")
    # The shared model uses one global WavLM vector for each video.
    return tensor.mean(dim=0, keepdim=True)


def cross_dataset_collate_fn(batch: Sequence[dict[str, Any]]):
    visual_list = [
        _normalize_visual_feature(item.get("visual_features", item.get("features")), 768)
        for item in batch
    ]
    audio_list = [_normalize_audio_feature(item["audio_features"], 1024) for item in batch]
    labels = torch.tensor([_label_value(item["label"]) for item in batch], dtype=torch.long)
    sample_ids = [f"{item['domain']}:{_sample_id(item)}" for item in batch]
    return visual_list, audio_list, labels, sample_ids


def _labels_from_dataset(dataset: Dataset) -> list[int]:
    if isinstance(dataset, DomainDataset):
        return _labels_from_dataset(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            labels.extend(_labels_from_dataset(child))
        return labels
    if isinstance(dataset, IndexedDataset):
        labels = []
        for child, index in dataset.references:
            if isinstance(child, DOLOSDataset):
                labels.append(int(child.samples[index][1]))
            elif isinstance(child, SEUMLDFineGrainedDataset):
                labels.append(int(child.samples[index][2]))
            else:
                labels.append(_label_value(child[index]["label"]))
        return labels
    if isinstance(dataset, DOLOSDataset):
        return [int(label) for _, label, _ in dataset.samples]
    if isinstance(dataset, SEUMLDFineGrainedDataset):
        return [int(label) for _, _, label, _ in dataset.samples]
    if isinstance(dataset, RealLifeLOOCVDataset):
        return [int(sample["label"]) for sample in dataset.current_split_data]
    return [_label_value(dataset[index]["label"]) for index in range(len(dataset))]


def _print_dataset_summary(dataset: DomainDataset) -> None:
    counts = Counter(_labels_from_dataset(dataset))
    print(
        f"  - {dataset.domain:<8}: samples={len(dataset):4d}, "
        f"truth={counts.get(0, 0):4d}, lie={counts.get(1, 0):4d}"
    )


def build_full_dolos_dataset() -> DomainDataset:
    args = DOLOSArgs()
    partitions = [
        DOLOSDataset(
            args.feature_root,
            args.fold_path,
            fold_index,
            split="test",
            audio_feature_root=args.audio_feature_root,
            modality="both",
            audio_dim=args.audio_dim,
        )
        for fold_index in range(3)
    ]

    references: list[tuple[Dataset, int]] = []
    seen_names: set[str] = set()
    for partition in partitions:
        for index, (_, _, video_name) in enumerate(partition.samples):
            if video_name not in seen_names:
                seen_names.add(video_name)
                references.append((partition, index))
    return DomainDataset("dolos", IndexedDataset(references))


def build_full_seumld_dataset() -> DomainDataset:
    args = SEUMLDArgs()
    partitions = [
        SEUMLDFineGrainedDataset(
            args.feature_root,
            args.label_path,
            args.fold_path,
            fold_index,
            split="test",
            audio_feature_root=args.audio_feature_root,
            modality="both",
            audio_dim=args.audio_dim,
        )
        for fold_index in range(5)
    ]

    references: list[tuple[Dataset, int]] = []
    seen_names: set[str] = set()
    for partition in partitions:
        for index, (_, _, _, video_name) in enumerate(partition.samples):
            if video_name not in seen_names:
                seen_names.add(video_name)
                references.append((partition, index))
    return DomainDataset("seumld", IndexedDataset(references))


def build_full_reallife_dataset() -> DomainDataset:
    args = RealLifeArgs()
    nested_data = RealLifeLOOCVDataset._load_data_from_pickle(args.feature_path)
    samples = RealLifeLOOCVDataset._flatten_data(nested_data)
    dataset = RealLifeLOOCVDataset(
        samples,
        original_indices=list(range(len(samples))),
        audio_feature_root=args.audio_feature_root,
        modality="both",
        audio_dim=args.audio_dim,
    )
    return DomainDataset("reallife", dataset)


def build_full_dataset(name: str) -> DomainDataset:
    builders = {
        "dolos": build_full_dolos_dataset,
        "seumld": build_full_seumld_dataset,
        "reallife": build_full_reallife_dataset,
    }
    return builders[name]()


def _make_domain_balanced_sampler(source_datasets: Sequence[DomainDataset], seed: int):
    weights: list[float] = []
    for dataset in source_datasets:
        if len(dataset) == 0:
            raise ValueError(f"Source dataset {dataset.domain} is empty")
        weights.extend([1.0 / len(dataset)] * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def _make_class_weights(source_datasets: Sequence[DomainDataset], device: torch.device) -> torch.Tensor:
    counts = Counter()
    for dataset in source_datasets:
        counts.update(_labels_from_dataset(dataset))
    total = counts[0] + counts[1]
    weights = [total / max(2 * counts[label], 1) for label in (0, 1)]
    print(f"  Class weights: truth={weights[0]:.4f}, lie={weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _move_feature_list(features: Iterable[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [feature.to(device, non_blocking=True) for feature in features]


def weak_instance_loss(
    instance_logits_list: Sequence[torch.Tensor],
    labels: torch.Tensor,
    topk_ratio: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for instance_logits, label in zip(instance_logits_list, labels):
        if instance_logits.numel() == 0:
            continue

        if int(label.item()) == 0:
            pseudo_labels = torch.zeros(
                instance_logits.size(0),
                dtype=torch.long,
                device=instance_logits.device,
            )
            losses.append(F.cross_entropy(instance_logits, pseudo_labels))
        else:
            num_pos = max(1, int(round(instance_logits.size(0) * topk_ratio)))
            lie_scores = F.softmax(instance_logits.detach(), dim=-1)[:, 1]
            top_indices = torch.topk(lie_scores, k=min(num_pos, instance_logits.size(0))).indices
            pseudo_labels = torch.ones(
                top_indices.size(0),
                dtype=torch.long,
                device=instance_logits.device,
            )
            losses.append(F.cross_entropy(instance_logits[top_indices], pseudo_labels))

    if not losses:
        return torch.tensor(0.0, device=labels.device)
    return torch.stack(losses).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    options: ExperimentOptions,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_ortho = 0.0
    total_instance = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{options.epochs} training")
    for visual_list, audio_list, labels, _ in progress:
        visual_list = _move_feature_list(visual_list, device)
        audio_list = _move_feature_list(audio_list, device)
        labels = labels.to(device, non_blocking=True)

        outputs = model(visual_list, audio_list, bag_labels=labels)
        ce_loss = criterion(outputs["logits"], labels)
        ortho_loss = outputs.get("ortho_loss", torch.tensor(0.0, device=device))
        instance_loss = torch.tensor(0.0, device=device)
        if options.use_instance_loss and "instance_logits_list" in outputs:
            instance_loss = weak_instance_loss(
                outputs["instance_logits_list"],
                labels,
                options.positive_instance_topk_ratio,
            )
        loss = (
            ce_loss
            + options.ortho_weight * ortho_loss
            + options.instance_loss_weight * instance_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_ortho += ortho_loss.item() * batch_size
        total_instance += instance_loss.item() * batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return {
        "loss": total_loss / max(total_samples, 1),
        "ce_loss": total_ce / max(total_samples, 1),
        "ortho_loss": total_ortho / max(total_samples, 1),
        "instance_loss": total_instance / max(total_samples, 1),
    }


def evaluate_target(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[float] = []
    ids_all: list[str] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for visual_list, audio_list, labels, sample_ids in tqdm(loader, desc="Held-out target test"):
            visual_list = _move_feature_list(visual_list, device)
            audio_list = _move_feature_list(audio_list, device)
            labels = labels.to(device, non_blocking=True)
            logits = model(visual_list, audio_list, bag_labels=None)["logits"]
            loss = criterion(logits, labels)
            probabilities = F.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities > 0.5).long()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            ids_all.extend(sample_ids)

    try:
        auc = roc_auc_score(labels_all, probabilities_all)
    except ValueError:
        auc = 0.0
    matrix = confusion_matrix(labels_all, predictions_all, labels=[0, 1])
    metrics = {
        "test_samples": total_samples,
        "test_loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision_score(labels_all, predictions_all, zero_division=0),
        "recall": recall_score(labels_all, predictions_all, zero_division=0),
        "f1": f1_score(labels_all, predictions_all, zero_division=0),
        "auc": auc,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels_all,
            predictions_all,
            target_names=["Truth (0)", "Lie (1)"],
            zero_division=0,
        ),
    }
    predictions = [
        {"id": sample_id, "label": label, "prediction": prediction, "lie_probability": probability}
        for sample_id, label, prediction, probability in zip(
            ids_all, labels_all, predictions_all, probabilities_all
        )
    ]
    return metrics, predictions


def _save_results(
    target_name: str,
    source_names: Sequence[str],
    model: nn.Module,
    options: ExperimentOptions,
    metrics: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
) -> None:
    experiment_dir = Path(options.output_dir) / f"{'_'.join(source_names)}_to_{target_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "target_dataset": target_name,
            "source_datasets": list(source_names),
            "model_state_dict": model.state_dict(),
            "options": asdict(options),
            "test_metrics": metrics,
        },
        experiment_dir / "final_model.pth.tar",
    )
    with (experiment_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (experiment_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target_dataset": target_name,
                "source_datasets": list(source_names),
                "options": asdict(options),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with (experiment_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "prediction", "lie_probability"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  Saved results to: {experiment_dir}")


def run_cross_dataset_experiment(target_name: str, options: ExperimentOptions | None = None):
    """Train on the other two datasets and test once on the complete target dataset."""

    if target_name not in DATASET_NAMES:
        raise ValueError(f"Unknown target dataset: {target_name}")
    options = options or ExperimentOptions()
    source_names = [name for name in DATASET_NAMES if name != target_name]
    set_random_seed(options.seed)

    print("\n" + "=" * 80)
    print(f"Sources: {' + '.join(source_names)}")
    print(f"Held-out test dataset: {target_name}")
    print("Protocol: target samples are used only for the final test")
    print(
        "Visual self-attention: "
        f"enabled={options.use_visual_self_attn}, "
        f"heads={options.visual_self_attn_heads}, "
        f"layers={options.visual_self_attn_layers}, "
        f"dropout={options.visual_self_attn_dropout}"
    )
    print(
        "Weak instance loss: "
        f"enabled={options.use_instance_loss}, "
        f"weight={options.instance_loss_weight}, "
        f"positive_topk_ratio={options.positive_instance_topk_ratio}"
    )
    print("=" * 80)

    source_datasets = [build_full_dataset(name) for name in source_names]
    target_dataset = build_full_dataset(target_name)
    print("\nDataset summary")
    for dataset in [*source_datasets, target_dataset]:
        _print_dataset_summary(dataset)
    if target_name == "reallife":
        print(f"  Real-Life held-out test coverage: all {len(target_dataset)} samples")

    source_dataset = ConcatDataset(source_datasets)
    sampler = _make_domain_balanced_sampler(source_datasets, options.seed)
    train_loader = DataLoader(
        source_dataset,
        batch_size=options.batch_size,
        sampler=sampler,
        num_workers=options.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=cross_dataset_collate_fn,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=cross_dataset_collate_fn,
    )

    if options.dry_run:
        visual_list, audio_list, labels, sample_ids = next(iter(train_loader))
        target_visual_list, target_audio_list, target_labels, target_sample_ids = next(iter(target_loader))
        dry_run_model = LieDetection(options).cpu().eval()
        with torch.no_grad():
            train_logits = dry_run_model(visual_list, audio_list, bag_labels=None)["logits"]
            target_logits = dry_run_model(target_visual_list, target_audio_list, bag_labels=None)["logits"]
        print("\nDry run: loaders built successfully")
        print(f"  Train batches: {len(train_loader)}, held-out test batches: {len(target_loader)}")
        print(
            f"  First train batch: visual={tuple(visual_list[0].shape)}, "
            f"audio={tuple(audio_list[0].shape)}, logits={tuple(train_logits.shape)}"
        )
        print(f"  Train labels: {tuple(labels.shape)}, first id: {sample_ids[0]}")
        print(
            f"  First target batch: visual={tuple(target_visual_list[0].shape)}, "
            f"audio={tuple(target_audio_list[0].shape)}, logits={tuple(target_logits.shape)}"
        )
        print(f"  Target labels: {tuple(target_labels.shape)}, first id: {target_sample_ids[0]}")
        return {"target_dataset": target_name, "test_samples": len(target_dataset), "dry_run": True}

    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Pass --device cpu for a CPU run.")
    device = torch.device(options.device)
    model = LieDetection(options).to(device)
    criterion = nn.CrossEntropyLoss(weight=_make_class_weights(source_datasets, device))
    optimizer = optim.SGD(
        model.parameters(),
        lr=options.lr,
        momentum=options.momentum,
        weight_decay=options.weight_decay,
    )

    print("\nTraining without target-domain validation")
    for epoch in range(options.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, options, epoch)
        print(
            f"  Epoch {epoch + 1:03d}: loss={train_metrics['loss']:.4f}, "
            f"ce={train_metrics['ce_loss']:.4f}, "
            f"inst={train_metrics['instance_loss']:.4f}, "
            f"ortho={train_metrics['ortho_loss']:.4f}"
        )

    print("\nFinal held-out target evaluation")
    metrics, predictions = evaluate_target(model, target_loader, criterion, device)
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print(metrics["classification_report"])
    _save_results(target_name, source_names, model, options, metrics, predictions)
    return metrics


def run_dolos_as_test(options: ExperimentOptions | None = None):
    """Train on SEUMLD + Real-Life, then test on the complete DOLOS dataset."""

    return run_cross_dataset_experiment("dolos", options)


def run_seumld_as_test(options: ExperimentOptions | None = None):
    """Train on DOLOS + Real-Life, then test on the complete SEUMLD dataset."""

    return run_cross_dataset_experiment("seumld", options)


def run_reallife_as_test(options: ExperimentOptions | None = None):
    """Train on DOLOS + SEUMLD, then test on all Real-Life samples."""

    return run_cross_dataset_experiment("reallife", options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=(*DATASET_NAMES, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--low-dim", type=int, default=64)
    parser.add_argument("--proto-m", type=float, default=0.9)
    parser.add_argument("--ortho-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", default="exper_model/cross_dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-visual-self-attn", action="store_true")
    parser.add_argument("--visual-self-attn-heads", type=int, default=4)
    parser.add_argument("--visual-self-attn-layers", type=int, default=1)
    parser.add_argument("--visual-self-attn-dropout", type=float, default=0.1)
    parser.add_argument("--no-instance-loss", dest="use_instance_loss", action="store_false")
    parser.add_argument("--instance-loss-weight", type=float, default=0.1)
    parser.add_argument("--positive-instance-topk-ratio", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_instance_loss=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = ExperimentOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        low_dim=args.low_dim,
        proto_m=args.proto_m,
        ortho_weight=args.ortho_weight,
        output_dir=args.output_dir,
        device=args.device,
        dry_run=args.dry_run,
        use_visual_self_attn=args.use_visual_self_attn,
        visual_self_attn_heads=args.visual_self_attn_heads,
        visual_self_attn_layers=args.visual_self_attn_layers,
        visual_self_attn_dropout=args.visual_self_attn_dropout,
        use_instance_loss=args.use_instance_loss,
        instance_loss_weight=args.instance_loss_weight,
        positive_instance_topk_ratio=args.positive_instance_topk_ratio,
    )
    targets = DATASET_NAMES if args.target == "all" else (args.target,)
    for target_name in targets:
        run_cross_dataset_experiment(target_name, options)


if __name__ == "__main__":
    main()
