import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.dolos import Args as DolosArgs
from configs.seumld import Args as SeumldArgs
from datasets.dataloaderFactory import create_dolos_dataloaders, create_seumld_dataloaders
from datasets.dolos import dolos_collate_fn
from datasets.seumld import finegrained_collate_fn, seumld_multimodal_collate_fn
from models.final import LieDetection
from utils.ins_loss import adjust_learning_rate
from utils.train import train_new_epo, train_seumld_epoch
from utils.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fold-internal validation calibrated training")
    parser.add_argument("--dataset", choices=("dolos", "seumld"), required=True)
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--folds", default="")
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-seed", type=int, default=2026)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--threshold-shrink-to-half", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=("acc_f1", "f1_acc"), default="acc_f1")
    parser.add_argument(
        "--checkpoint-metric",
        choices=("selection", "auc_acc", "acc_auc", "loss"),
        default="selection",
    )
    parser.add_argument("--f1-weight", type=float, default=0.25)
    parser.add_argument("--retrain-full", action="store_true")
    parser.add_argument("--snapshot-ensemble-k", type=int, default=1)
    parser.add_argument("--modality", choices=("visual", "audio", "both"), default=None)
    parser.add_argument("--aggr-method", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--low-dim", type=int, default=None)
    parser.add_argument("--use-logit-topk-pooling", action="store_true")
    parser.add_argument("--logit-topk-pooling-ratio", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--lie-class-weight", type=float, default=None)
    parser.add_argument("--use-logit-margin-regularization", action="store_true")
    parser.add_argument("--logit-margin-weight", type=float, default=None)
    parser.add_argument("--logit-margin-target", type=float, default=None)
    parser.add_argument("--logit-margin-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-batch-rank-loss", action="store_true")
    parser.add_argument("--batch-rank-loss-weight", type=float, default=None)
    parser.add_argument("--batch-rank-margin", type=float, default=None)
    parser.add_argument("--batch-rank-warmup-epochs", type=int, default=None)
    parser.add_argument("--visual-logit-ensemble-weight", type=float, default=None)
    parser.add_argument("--audio-residual-drop-prob", type=float, default=None)
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=None)
    parser.add_argument("--use-visual-aux-loss", action="store_true")
    parser.add_argument("--visual-aux-loss-weight", type=float, default=None)
    parser.add_argument("--visual-aux-loss-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-fusion-consistency-loss", action="store_true")
    parser.add_argument("--fusion-consistency-loss-weight", type=float, default=None)
    parser.add_argument("--fusion-consistency-temperature", type=float, default=None)
    parser.add_argument("--fusion-consistency-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-mil-evidence-loss", action="store_true")
    parser.add_argument("--mil-evidence-loss-weight", type=float, default=None)
    parser.add_argument("--mil-evidence-topk-ratio", type=float, default=None)
    parser.add_argument("--mil-evidence-rank-weight", type=float, default=None)
    parser.add_argument("--mil-evidence-rank-margin", type=float, default=None)
    parser.add_argument("--mil-evidence-warmup-epochs", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser.parse_args()


def labels_from_dataset(dataset):
    labels = []
    for sample in dataset.samples:
        if len(sample) == 3:
            labels.append(int(sample[1]))
        else:
            labels.append(int(sample[2]))
    return np.asarray(labels, dtype=np.int64)


def stratified_split_indices(labels, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in sorted(np.unique(labels)):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        val_count = max(1, int(round(len(label_indices) * val_ratio)))
        val_indices.extend(label_indices[:val_count].tolist())
        train_indices.extend(label_indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_optimizer(args, model):
    if args.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def class_weights_from_labels(labels, dataset_name, args):
    count0 = int((labels == 0).sum())
    count1 = int((labels == 1).sum())
    if count0 > 0 and count1 > 0:
        weight1 = count0 / count1
        if dataset_name == "dolos":
            weight1 = getattr(args, "lie_class_weight", max(weight1, 1.0))
        else:
            weight1 = min(max(weight1, 1.0), 3.0)
    else:
        weight1 = 1.5
    return torch.tensor([1.0, float(weight1)], device="cuda")


def to_cuda(features_list):
    if features_list is None:
        return None
    return [item.cuda(non_blocking=True) if item is not None else None for item in features_list]


def collect_probs(model, loader, criterion):
    model.eval()
    probs = []
    labels = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for visual_list, audio_list, bag_labels, _ in loader:
            visual_list = to_cuda(visual_list)
            audio_list = to_cuda(audio_list)
            bag_labels = bag_labels.cuda(non_blocking=True)
            outputs = model(visual_list, audio_list, bag_labels=None)
            logits = outputs["logits"]
            total_loss += criterion(logits, bag_labels).item()
            batches += 1
            probs.append(F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
            labels.append(bag_labels.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), total_loss / max(batches, 1)


def metrics_at_threshold(probs, labels, threshold, loss):
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
        "val_loss": float(loss),
        "threshold": float(threshold),
    }


def calibrate_threshold(probs, labels, loss, run_args):
    best = None
    best_metrics = None
    thresholds = np.arange(
        run_args.threshold_min,
        run_args.threshold_max + 1e-12,
        run_args.threshold_step,
    )
    for threshold in thresholds:
        item = metrics_at_threshold(probs, labels, threshold, loss)
        if run_args.selection_metric == "f1_acc":
            key = (item["f1"], item["accuracy"], item["auc"], -abs(threshold - 0.5))
        else:
            key = (
                item["accuracy"] + run_args.f1_weight * item["f1"],
                item["accuracy"],
                item["f1"],
                item["auc"],
                -abs(threshold - 0.5),
            )
        if best is None or key > best:
            best = key
            best_metrics = item
    return best_metrics


def maybe_shrink_threshold(threshold, run_args):
    shrink = float(getattr(run_args, "threshold_shrink_to_half", 0.0))
    shrink = min(max(shrink, 0.0), 1.0)
    return float(threshold * (1.0 - shrink) + 0.5 * shrink)


def checkpoint_key(val_metrics, run_args):
    if run_args.checkpoint_metric == "auc_acc":
        return (
            val_metrics["auc"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            -val_metrics["val_loss"],
        )
    if run_args.checkpoint_metric == "acc_auc":
        return (
            val_metrics["accuracy"],
            val_metrics["auc"],
            val_metrics["f1"],
            -val_metrics["val_loss"],
        )
    if run_args.checkpoint_metric == "loss":
        return (
            -val_metrics["val_loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
            val_metrics["f1"],
        )
    return (
        val_metrics["accuracy"] + run_args.f1_weight * val_metrics["f1"],
        val_metrics["accuracy"],
        val_metrics["f1"],
        val_metrics["auc"],
    )


def apply_common_off(args):
    for name in (
        "use_visual_self_attn",
        "use_instance_loss",
        "use_topk_proto_update",
        "use_conservative_topk_proto_update",
        "use_proto_memory_queue",
        "use_cluster_topk_mean_pooling",
        "use_cluster_topk_weighted_pooling",
        "use_cluster_margin_topk_pooling",
        "use_logit_warm_assignment",
        "use_dual_score_topk_pooling",
        "use_logit_topk_pooling",
        "use_enhanced_fusion_gate",
        "use_bag_proto_logits",
        "use_visual_logit_ensemble",
        "use_audio_residual_drop",
        "use_visual_aux_loss",
        "use_fusion_consistency_loss",
        "use_mil_evidence_loss",
        "use_proto_loop_consistency",
        "use_logit_margin_regularization",
        "use_batch_rank_loss",
    ):
        if hasattr(args, name):
            setattr(args, name, False)


def apply_preset(args, preset):
    apply_common_off(args)
    if preset == "baseline":
        return
    if preset in ("cluster", "dolos_cluster_vens_drop", "seu_cluster_vsa", "seu_cluster_vsa_inst"):
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
    if preset == "dolos_cluster_vens_drop":
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
    elif preset == "dolos_cluster_vens_drop_vaux":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_visual_aux_loss = True
        args.visual_aux_loss_weight = 0.2
    elif preset == "dolos_cluster_vens_drop_vaux_fcons":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_visual_aux_loss = True
        args.visual_aux_loss_weight = 0.2
        args.use_fusion_consistency_loss = True
        args.fusion_consistency_loss_weight = 0.05
        args.fusion_consistency_temperature = 2.0
    elif preset == "dolos_cluster_vens_drop_mil":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_mil_evidence_loss = True
        args.mil_evidence_loss_weight = 0.1
        args.mil_evidence_topk_ratio = 0.25
        args.mil_evidence_rank_weight = 0.05
        args.mil_evidence_rank_margin = 0.5
    elif preset == "dolos_cluster_vens_drop_mil_vaux":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_mil_evidence_loss = True
        args.mil_evidence_loss_weight = 0.1
        args.mil_evidence_topk_ratio = 0.25
        args.mil_evidence_rank_weight = 0.05
        args.mil_evidence_rank_margin = 0.5
        args.use_visual_aux_loss = True
        args.visual_aux_loss_weight = 0.2
    elif preset == "dolos_cluster_vens_drop_mil_calib":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_mil_evidence_loss = True
        args.mil_evidence_loss_weight = 0.2
        args.mil_evidence_topk_ratio = 0.25
        args.mil_evidence_rank_weight = 0.05
        args.mil_evidence_rank_margin = 0.5
        args.label_smoothing = 0.05
        args.use_logit_margin_regularization = True
        args.logit_margin_weight = 0.03
        args.logit_margin_target = 2.0
        args.logit_margin_warmup_epochs = 10
    elif preset == "dolos_logit_topk_mil":
        args.use_logit_topk_pooling = True
        args.logit_topk_pooling_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_mil_evidence_loss = True
        args.mil_evidence_loss_weight = 0.1
        args.mil_evidence_topk_ratio = 0.25
        args.mil_evidence_rank_weight = 0.05
        args.mil_evidence_rank_margin = 0.5
    elif preset == "dolos_vens09":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.9
    elif preset == "seu_cluster_vsa":
        args.use_visual_self_attn = True
        args.visual_self_attn_heads = 4
        args.visual_self_attn_layers = 1
        args.use_instance_loss = False
    elif preset == "seu_cluster_vsa_inst":
        args.use_visual_self_attn = True
        args.visual_self_attn_heads = 4
        args.visual_self_attn_layers = 1
        args.use_instance_loss = True
        args.instance_loss_weight = 0.05
    elif preset == "seu_cluster_margin":
        args.use_cluster_topk_mean_pooling = True
        args.use_cluster_margin_topk_pooling = True
        args.cluster_margin_topk_ratio = 0.5
        args.use_instance_loss = False
    elif preset == "seu_vens_drop":
        args.use_cluster_topk_mean_pooling = True
        args.cluster_topk_mean_ratio = 0.5
        args.use_visual_logit_ensemble = True
        args.visual_logit_ensemble_weight = 0.6
        args.use_audio_residual_drop = True
        args.audio_residual_drop_prob = 0.3
        args.use_instance_loss = False
    else:
        raise ValueError(f"Unknown preset: {preset}")


def apply_overrides(args, run_args):
    if run_args.epochs is not None:
        args.epochs = run_args.epochs
    if run_args.batch_size is not None:
        args.batch_size = run_args.batch_size
    if run_args.num_workers is not None:
        args.num_workers = run_args.num_workers
    if run_args.disable_tqdm:
        args.disable_tqdm = True
    if run_args.lr is not None:
        args.lr = run_args.lr
    if run_args.weight_decay is not None:
        args.weight_decay = run_args.weight_decay
    if run_args.optimizer is not None:
        args.optimizer = run_args.optimizer
    if run_args.seed is not None:
        args.seed = run_args.seed
    if run_args.num_runs is not None:
        args.num_runs = run_args.num_runs
    if run_args.modality is not None:
        args.modality = run_args.modality
    if run_args.aggr_method is not None:
        args.aggr_method = run_args.aggr_method
    if run_args.hidden_dim is not None:
        args.hidden_dim = run_args.hidden_dim
    if run_args.low_dim is not None:
        args.low_dim = run_args.low_dim
    if run_args.use_logit_topk_pooling:
        args.use_logit_topk_pooling = True
    if run_args.logit_topk_pooling_ratio is not None:
        args.logit_topk_pooling_ratio = run_args.logit_topk_pooling_ratio
        args.use_logit_topk_pooling = True
    if run_args.label_smoothing is not None:
        args.label_smoothing = run_args.label_smoothing
    if run_args.lie_class_weight is not None:
        args.lie_class_weight = run_args.lie_class_weight
    if run_args.use_logit_margin_regularization:
        args.use_logit_margin_regularization = True
    if run_args.logit_margin_weight is not None:
        args.logit_margin_weight = run_args.logit_margin_weight
        args.use_logit_margin_regularization = True
    if run_args.logit_margin_target is not None:
        args.logit_margin_target = run_args.logit_margin_target
        args.use_logit_margin_regularization = True
    if run_args.logit_margin_warmup_epochs is not None:
        args.logit_margin_warmup_epochs = run_args.logit_margin_warmup_epochs
    if run_args.use_batch_rank_loss:
        args.use_batch_rank_loss = True
    if run_args.batch_rank_loss_weight is not None:
        args.batch_rank_loss_weight = run_args.batch_rank_loss_weight
        args.use_batch_rank_loss = True
    if run_args.batch_rank_margin is not None:
        args.batch_rank_margin = run_args.batch_rank_margin
    if run_args.batch_rank_warmup_epochs is not None:
        args.batch_rank_warmup_epochs = run_args.batch_rank_warmup_epochs
    if run_args.visual_logit_ensemble_weight is not None:
        args.visual_logit_ensemble_weight = run_args.visual_logit_ensemble_weight
        args.use_visual_logit_ensemble = True
    if run_args.audio_residual_drop_prob is not None:
        args.audio_residual_drop_prob = run_args.audio_residual_drop_prob
        args.use_audio_residual_drop = True
    if run_args.cluster_topk_mean_ratio is not None:
        args.cluster_topk_mean_ratio = run_args.cluster_topk_mean_ratio
        args.use_cluster_topk_mean_pooling = True
    if run_args.use_visual_aux_loss:
        args.use_visual_aux_loss = True
    if run_args.visual_aux_loss_weight is not None:
        args.visual_aux_loss_weight = run_args.visual_aux_loss_weight
        args.use_visual_aux_loss = True
    if run_args.visual_aux_loss_warmup_epochs is not None:
        args.visual_aux_loss_warmup_epochs = run_args.visual_aux_loss_warmup_epochs
    if run_args.use_fusion_consistency_loss:
        args.use_fusion_consistency_loss = True
    if run_args.fusion_consistency_loss_weight is not None:
        args.fusion_consistency_loss_weight = run_args.fusion_consistency_loss_weight
        args.use_fusion_consistency_loss = True
    if run_args.fusion_consistency_temperature is not None:
        args.fusion_consistency_temperature = run_args.fusion_consistency_temperature
    if run_args.fusion_consistency_warmup_epochs is not None:
        args.fusion_consistency_warmup_epochs = run_args.fusion_consistency_warmup_epochs
    if run_args.use_mil_evidence_loss:
        args.use_mil_evidence_loss = True
    if run_args.mil_evidence_loss_weight is not None:
        args.mil_evidence_loss_weight = run_args.mil_evidence_loss_weight
        args.use_mil_evidence_loss = True
    if run_args.mil_evidence_topk_ratio is not None:
        args.mil_evidence_topk_ratio = run_args.mil_evidence_topk_ratio
        args.use_mil_evidence_loss = True
    if run_args.mil_evidence_rank_weight is not None:
        args.mil_evidence_rank_weight = run_args.mil_evidence_rank_weight
        args.use_mil_evidence_loss = True
    if run_args.mil_evidence_rank_margin is not None:
        args.mil_evidence_rank_margin = run_args.mil_evidence_rank_margin
    if run_args.mil_evidence_warmup_epochs is not None:
        args.mil_evidence_warmup_epochs = run_args.mil_evidence_warmup_epochs


def build_loaders(dataset_name, args, fold_idx):
    if dataset_name == "dolos":
        _, test_loader, train_dataset, _ = create_dolos_dataloaders(
            feature_root=args.feature_root,
            fold_path=args.fold_path,
            collate_fn=dolos_collate_fn,
            fold_index=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_feature_root=getattr(args, "audio_feature_root", None),
            modality=getattr(args, "modality", "both"),
            audio_dim=getattr(args, "audio_dim", 1024),
        )
        collate_fn = dolos_collate_fn
    else:
        collate_fn = finegrained_collate_fn if args.modality == "visual" else seumld_multimodal_collate_fn
        _, test_loader, train_dataset, _ = create_seumld_dataloaders(
            feature_root=args.feature_root,
            label_path=args.label_path,
            fold_path=args.fold_path,
            collate_fn=collate_fn,
            fold_index=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_feature_root=getattr(args, "audio_feature_root", None),
            modality=getattr(args, "modality", "both"),
            audio_dim=getattr(args, "audio_dim", 1024),
        )
    return train_dataset, test_loader, collate_fn


def run_fold(dataset_name, base_args, run_args, fold_idx, log):
    seed = base_args.seed + fold_idx
    set_random_seed(seed)
    train_dataset, test_loader, collate_fn = build_loaders(dataset_name, base_args, fold_idx)
    labels = labels_from_dataset(train_dataset)
    subtrain_indices, val_indices = stratified_split_indices(
        labels,
        run_args.val_ratio,
        run_args.val_seed + fold_idx,
    )
    train_subset = Subset(train_dataset, subtrain_indices)
    val_subset = Subset(train_dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=base_args.batch_size,
        shuffle=True,
        num_workers=base_args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=base_args.batch_size,
        shuffle=False,
        num_workers=base_args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    train_labels = labels[subtrain_indices]
    class_weights = class_weights_from_labels(train_labels, dataset_name, base_args)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=getattr(base_args, "label_smoothing", 0.0),
    )
    model = LieDetection(base_args).cuda()
    optimizer = build_optimizer(base_args, model)

    log(
        f"Fold {fold_idx + 1}: train={len(train_subset)} val={len(val_subset)} "
        f"test_batches={len(test_loader)} class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}"
    )
    best_val = None
    best_state = None
    best_epoch = 0
    stale = 0
    last_eval_epoch = 0
    fold_val_history = []
    fold_eval_records = []

    train_fn = train_new_epo if dataset_name == "dolos" else train_seumld_epoch
    for epoch in range(base_args.epochs):
        adjust_learning_rate(base_args, optimizer, epoch)
        train_metrics = train_fn(model, train_loader, criterion, optimizer, epoch, base_args)
        should_eval = (
            epoch + 1 == base_args.epochs
            or epoch + 1 <= run_args.min_epochs
            or (epoch + 1) % run_args.eval_interval == 0
        )
        if not should_eval:
            continue
        eval_epoch = epoch + 1
        elapsed_since_eval = max(eval_epoch - last_eval_epoch, 1)
        last_eval_epoch = eval_epoch
        val_probs, val_labels, val_loss = collect_probs(model, val_loader, criterion)
        val_metrics = calibrate_threshold(val_probs, val_labels, val_loss, run_args)
        fold_val_history.append({"epoch": epoch + 1, **val_metrics})
        fold_eval_records.append(
            {
                "epoch": eval_epoch,
                "metrics": dict(val_metrics),
                "val_probs": val_probs.copy(),
                "val_labels": val_labels.copy(),
            }
        )
        key = checkpoint_key(val_metrics, run_args)
        old_key = None
        if best_val is not None:
            old_key = checkpoint_key(best_val, run_args)
        log(
            f"Fold {fold_idx + 1} Epoch {epoch + 1}: "
            f"train_loss={train_metrics.get('train_loss', 0.0):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_th={val_metrics['threshold']:.3f}"
        )
        if old_key is None or key > old_key:
            best_val = dict(val_metrics)
            best_epoch = eval_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += elapsed_since_eval
            if eval_epoch >= run_args.min_epochs and stale >= run_args.early_stop_patience:
                log(f"Fold {fold_idx + 1}: early stop at epoch {eval_epoch}, best_epoch={best_epoch}")
                break

    snapshot_k = max(1, int(getattr(run_args, "snapshot_ensemble_k", 1)))
    selected_records = sorted(
        fold_eval_records,
        key=lambda item: checkpoint_key(item["metrics"], run_args),
        reverse=True,
    )[:snapshot_k]
    selected_epochs = sorted({item["epoch"] for item in selected_records})
    if snapshot_k > 1 and selected_records:
        selected_val_probs = np.stack([item["val_probs"] for item in selected_records], axis=0)
        val_probs = selected_val_probs.mean(axis=0)
        val_labels = selected_records[0]["val_labels"]
        best_val = calibrate_threshold(val_probs, val_labels, 0.0, run_args)
        best_epoch = max(selected_epochs)
    elif best_state is not None:
        model.load_state_dict(best_state)
    eval_threshold = maybe_shrink_threshold(best_val["threshold"], run_args)

    if run_args.retrain_full:
        set_random_seed(seed)
        full_train_loader = DataLoader(
            train_dataset,
            batch_size=base_args.batch_size,
            shuffle=True,
            num_workers=base_args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        full_labels = labels
        full_class_weights = class_weights_from_labels(full_labels, dataset_name, base_args)
        full_criterion = nn.CrossEntropyLoss(
            weight=full_class_weights,
            label_smoothing=getattr(base_args, "label_smoothing", 0.0),
        )
        model = LieDetection(base_args).cuda()
        optimizer = build_optimizer(base_args, model)
        log(
            f"Fold {fold_idx + 1}: retrain_full epochs={best_epoch} "
            f"snapshots={selected_epochs} "
            f"class_weights={full_class_weights.detach().cpu().numpy().round(4).tolist()} "
            f"calib_th={best_val['threshold']:.3f} eval_th={eval_threshold:.3f}"
        )
        snapshot_test_probs = []
        snapshot_test_labels = None
        for retrain_epoch in range(best_epoch):
            adjust_learning_rate(base_args, optimizer, retrain_epoch)
            train_fn(model, full_train_loader, full_criterion, optimizer, retrain_epoch, base_args)
            epoch_number = retrain_epoch + 1
            if snapshot_k > 1 and epoch_number in selected_epochs:
                snap_probs, snap_labels, _ = collect_probs(model, test_loader, full_criterion)
                snapshot_test_probs.append(snap_probs)
                snapshot_test_labels = snap_labels
        criterion = full_criterion

    if run_args.retrain_full and snapshot_k > 1 and snapshot_test_probs:
        test_probs = np.stack(snapshot_test_probs, axis=0).mean(axis=0)
        test_labels = snapshot_test_labels
        test_loss = 0.0
    else:
        test_probs, test_labels, test_loss = collect_probs(model, test_loader, criterion)
    test_metrics = metrics_at_threshold(test_probs, test_labels, eval_threshold, test_loss)
    score_path = Path(base_args.exp_dir) / f"fold_{fold_idx + 1}_scores.npz"
    np.savez_compressed(
        score_path,
        val_probs=val_probs,
        val_labels=val_labels,
        test_probs=test_probs,
        test_labels=test_labels,
        threshold=np.asarray([eval_threshold], dtype=np.float32),
        calibration_threshold=np.asarray([best_val["threshold"]], dtype=np.float32),
        retrain_full=np.asarray([int(run_args.retrain_full)], dtype=np.int64),
        best_epoch=np.asarray([best_epoch], dtype=np.int64),
        snapshot_epochs=np.asarray(selected_epochs, dtype=np.int64),
    )
    history_path = Path(base_args.exp_dir) / f"fold_{fold_idx + 1}_val_history.csv"
    pd.DataFrame(fold_val_history).to_csv(history_path, index=False)
    log(
        f"Fold {fold_idx + 1} BEST: epoch={best_epoch} snapshots={selected_epochs} "
        f"val_acc={best_val['accuracy']:.4f} val_f1={best_val['f1']:.4f} "
        f"val_th={best_val['threshold']:.3f} eval_th={eval_threshold:.3f} | "
        f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1']:.4f} "
        f"test_auc={test_metrics['auc']:.4f} scores={score_path}"
    )
    return {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "snapshot_epochs": ",".join(str(item) for item in selected_epochs),
        "val_accuracy": best_val["accuracy"],
        "val_f1": best_val["f1"],
        "val_auc": best_val["auc"],
        "threshold": eval_threshold,
        "calibration_threshold": best_val["threshold"],
        "retrain_full": int(run_args.retrain_full),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }


def main():
    run_args = parse_args()
    base_args = DolosArgs() if run_args.dataset == "dolos" else SeumldArgs()
    apply_preset(base_args, run_args.preset)
    apply_overrides(base_args, run_args)
    base_args.exp_dir = os.path.join(
        "/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/val_calibrated",
        run_args.dataset,
        run_args.exp_name,
    )
    Path(base_args.exp_dir).mkdir(parents=True, exist_ok=True)

    log_path = Path(base_args.exp_dir) / "run.log"

    def log(message):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    if log_path.exists():
        log_path.unlink()
    log(f"dataset={run_args.dataset} preset={run_args.preset} exp={run_args.exp_name}")
    log(
        f"epochs={base_args.epochs} batch_size={base_args.batch_size} lr={base_args.lr} "
        f"optimizer={base_args.optimizer} val_ratio={run_args.val_ratio} "
        f"eval_interval={run_args.eval_interval} patience={run_args.early_stop_patience} "
        f"selection={run_args.selection_metric} checkpoint={run_args.checkpoint_metric} "
        f"retrain_full={run_args.retrain_full} snapshot_k={run_args.snapshot_ensemble_k} "
        f"threshold_shrink={run_args.threshold_shrink_to_half}"
    )
    log(
        f"components: vsa={getattr(base_args, 'use_visual_self_attn', False)} "
        f"inst={getattr(base_args, 'use_instance_loss', False)} "
        f"cluster={getattr(base_args, 'use_cluster_topk_mean_pooling', False)} "
        f"logit_topk={getattr(base_args, 'use_logit_topk_pooling', False)} "
        f"logit_topk_ratio={getattr(base_args, 'logit_topk_pooling_ratio', 0.0)} "
        f"vens={getattr(base_args, 'use_visual_logit_ensemble', False)} "
        f"vens_w={getattr(base_args, 'visual_logit_ensemble_weight', 0.0)} "
        f"ardrop={getattr(base_args, 'use_audio_residual_drop', False)} "
        f"ardrop_p={getattr(base_args, 'audio_residual_drop_prob', 0.0)} "
        f"vaux={getattr(base_args, 'use_visual_aux_loss', False)} "
        f"vaux_w={getattr(base_args, 'visual_aux_loss_weight', 0.0)} "
        f"fcons={getattr(base_args, 'use_fusion_consistency_loss', False)} "
        f"fcons_w={getattr(base_args, 'fusion_consistency_loss_weight', 0.0)} "
        f"fcons_t={getattr(base_args, 'fusion_consistency_temperature', 0.0)} "
        f"mil={getattr(base_args, 'use_mil_evidence_loss', False)} "
        f"mil_w={getattr(base_args, 'mil_evidence_loss_weight', 0.0)} "
        f"mil_topk={getattr(base_args, 'mil_evidence_topk_ratio', 0.0)} "
        f"mil_rank_w={getattr(base_args, 'mil_evidence_rank_weight', 0.0)} "
        f"ls={getattr(base_args, 'label_smoothing', 0.0)} "
        f"lmargin={getattr(base_args, 'use_logit_margin_regularization', False)} "
        f"lmargin_w={getattr(base_args, 'logit_margin_weight', 0.0)} "
        f"lmargin_t={getattr(base_args, 'logit_margin_target', 0.0)} "
        f"brank={getattr(base_args, 'use_batch_rank_loss', False)} "
        f"brank_w={getattr(base_args, 'batch_rank_loss_weight', 0.0)} "
        f"modality={getattr(base_args, 'modality', 'both')} aggr={getattr(base_args, 'aggr_method', '')}"
    )

    if run_args.folds:
        folds = [int(item) - 1 for item in run_args.folds.split(",") if item.strip()]
    else:
        folds = list(range(base_args.num_runs))

    results = []
    for fold_idx in folds:
        results.append(run_fold(run_args.dataset, base_args, run_args, fold_idx, log))

    df = pd.DataFrame(results)
    csv_path = Path(base_args.exp_dir) / "results.csv"
    df.to_csv(csv_path, index=False)
    log("\nPer-fold results:")
    log(df.round(4).to_string(index=False))
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    log("\nSummary:")
    for key in ("test_accuracy", "test_f1", "test_auc", "test_precision", "test_recall"):
        log(f"{key}: {mean[key]:.4f} +/- {std[key]:.4f}")
    log(f"results_csv={csv_path}")


if __name__ == "__main__":
    main()
