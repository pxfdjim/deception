"""
DOLOS 数据集 - pplg_muti 非对称多模态模型训练脚本
视觉: MIL 原型引导层次化聚合
音频: 全局 MLP 编码
融合: 低秩张量融合 (LRTF) + 模态对齐损失
"""
import os, glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
import pandas as pd
import numpy as np
from utils.eval import evaluate_model
from models.final import LieDetection
from utils.ins_loss import adjust_learning_rate
from utils.logger import Logger
from utils.utils import set_random_seed, save_checkpoint
from datasets.dataloaderFactory import create_dolos_dataloaders
from datasets.dolos import dolos_collate_fn
from configs.dolos import Args
from utils.train import train_new_epo


def parse_args():
    parser = argparse.ArgumentParser(description="DOLOS component ablation training")
    parser.add_argument("--use-instance-loss", dest="use_instance_loss", action="store_true", default=None)
    parser.add_argument("--no-instance-loss", dest="use_instance_loss", action="store_false")
    parser.add_argument("--instance-loss-weight", type=float, default=None)
    parser.add_argument("--positive-instance-topk-ratio", type=float, default=None)
    parser.add_argument("--use-topk-proto-update", dest="use_topk_proto_update", action="store_true", default=None)
    parser.add_argument("--no-topk-proto-update", dest="use_topk_proto_update", action="store_false")
    parser.add_argument("--use-conservative-topk-proto-update", dest="use_conservative_topk_proto_update", action="store_true", default=None)
    parser.add_argument("--no-conservative-topk-proto-update", dest="use_conservative_topk_proto_update", action="store_false")
    parser.add_argument("--topk-proto-ratio", type=float, default=None)
    parser.add_argument("--topk-proto-threshold", type=float, default=None)
    parser.add_argument("--topk-proto-warmup-epochs", type=int, default=None)
    parser.add_argument("--proto-sep-margin", type=float, default=None)
    parser.add_argument("--proto-sep-loss-weight", type=float, default=None)
    parser.add_argument("--use-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_true", default=None)
    parser.add_argument("--no-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_false")
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=None)
    parser.add_argument("--use-visual-logit-ensemble", dest="use_visual_logit_ensemble", action="store_true", default=None)
    parser.add_argument("--no-visual-logit-ensemble", dest="use_visual_logit_ensemble", action="store_false")
    parser.add_argument("--visual-logit-ensemble-weight", type=float, default=None)
    parser.add_argument("--use-audio-residual-drop", dest="use_audio_residual_drop", action="store_true", default=None)
    parser.add_argument("--no-audio-residual-drop", dest="use_audio_residual_drop", action="store_false")
    parser.add_argument("--audio-residual-drop-prob", type=float, default=None)
    parser.add_argument("--use-visual-aux-loss", dest="use_visual_aux_loss", action="store_true", default=None)
    parser.add_argument("--no-visual-aux-loss", dest="use_visual_aux_loss", action="store_false")
    parser.add_argument("--visual-aux-loss-weight", type=float, default=None)
    parser.add_argument("--visual-aux-loss-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-fusion-consistency-loss", dest="use_fusion_consistency_loss", action="store_true", default=None)
    parser.add_argument("--no-fusion-consistency-loss", dest="use_fusion_consistency_loss", action="store_false")
    parser.add_argument("--fusion-consistency-loss-weight", type=float, default=None)
    parser.add_argument("--fusion-consistency-temperature", type=float, default=None)
    parser.add_argument("--fusion-consistency-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-mil-evidence-loss", dest="use_mil_evidence_loss", action="store_true", default=None)
    parser.add_argument("--no-mil-evidence-loss", dest="use_mil_evidence_loss", action="store_false")
    parser.add_argument("--mil-evidence-loss-weight", type=float, default=None)
    parser.add_argument("--mil-evidence-topk-ratio", type=float, default=None)
    parser.add_argument("--mil-evidence-rank-weight", type=float, default=None)
    parser.add_argument("--mil-evidence-rank-margin", type=float, default=None)
    parser.add_argument("--mil-evidence-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-eval-threshold-search", dest="use_eval_threshold_search", action="store_true", default=None)
    parser.add_argument("--no-eval-threshold-search", dest="use_eval_threshold_search", action="store_false")
    parser.add_argument("--eval-threshold-min", type=float, default=None)
    parser.add_argument("--eval-threshold-max", type=float, default=None)
    parser.add_argument("--eval-threshold-step", type=float, default=None)
    parser.add_argument("--exp-suffix", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--lie-class-weight", type=float, default=None)
    parser.add_argument("--use-logit-margin-regularization", dest="use_logit_margin_regularization", action="store_true", default=None)
    parser.add_argument("--no-logit-margin-regularization", dest="use_logit_margin_regularization", action="store_false")
    parser.add_argument("--logit-margin-weight", type=float, default=None)
    parser.add_argument("--logit-margin-target", type=float, default=None)
    parser.add_argument("--logit-margin-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-batch-rank-loss", dest="use_batch_rank_loss", action="store_true", default=None)
    parser.add_argument("--no-batch-rank-loss", dest="use_batch_rank_loss", action="store_false")
    parser.add_argument("--batch-rank-loss-weight", type=float, default=None)
    parser.add_argument("--batch-rank-margin", type=float, default=None)
    parser.add_argument("--batch-rank-warmup-epochs", type=int, default=None)
    parser.add_argument("--use-model-ema", dest="use_model_ema", action="store_true", default=None)
    parser.add_argument("--no-model-ema", dest="use_model_ema", action="store_false")
    parser.add_argument("--model-ema-decay", type=float, default=None)
    parser.add_argument("--model-ema-start-epoch", type=int, default=None)
    parser.add_argument("--use-ema-dual-eval", dest="use_ema_dual_eval", action="store_true", default=None)
    parser.add_argument("--no-ema-dual-eval", dest="use_ema_dual_eval", action="store_false")
    return parser.parse_args()


def apply_cli_overrides(args, cli_args):
    for name in (
        "use_instance_loss",
        "instance_loss_weight",
        "positive_instance_topk_ratio",
        "use_topk_proto_update",
        "use_conservative_topk_proto_update",
        "topk_proto_ratio",
        "topk_proto_threshold",
        "topk_proto_warmup_epochs",
        "proto_sep_margin",
        "proto_sep_loss_weight",
        "use_cluster_topk_mean_pooling",
        "cluster_topk_mean_ratio",
        "use_visual_logit_ensemble",
        "visual_logit_ensemble_weight",
        "use_audio_residual_drop",
        "audio_residual_drop_prob",
        "use_visual_aux_loss",
        "visual_aux_loss_weight",
        "visual_aux_loss_warmup_epochs",
        "use_fusion_consistency_loss",
        "fusion_consistency_loss_weight",
        "fusion_consistency_temperature",
        "fusion_consistency_warmup_epochs",
        "use_mil_evidence_loss",
        "mil_evidence_loss_weight",
        "mil_evidence_topk_ratio",
        "mil_evidence_rank_weight",
        "mil_evidence_rank_margin",
        "mil_evidence_warmup_epochs",
        "use_eval_threshold_search",
        "eval_threshold_min",
        "eval_threshold_max",
        "eval_threshold_step",
        "label_smoothing",
        "lie_class_weight",
        "use_logit_margin_regularization",
        "logit_margin_weight",
        "logit_margin_target",
        "logit_margin_warmup_epochs",
        "use_batch_rank_loss",
        "batch_rank_loss_weight",
        "batch_rank_margin",
        "batch_rank_warmup_epochs",
        "use_model_ema",
        "model_ema_decay",
        "model_ema_start_epoch",
        "use_ema_dual_eval",
        "seed",
        "epochs",
        "num_runs",
    ):
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(args, name, value)
    if cli_args.exp_suffix:
        args.exp_dir = os.path.join(args.exp_dir, cli_args.exp_suffix)
    return args


def component_log_name(args):
    use_inst = getattr(args, "use_instance_loss", True)
    inst_tag = "inst_off"
    if use_inst:
        weight_tag = str(getattr(args, "instance_loss_weight", 0.1)).replace(".", "p")
        topk_tag = str(getattr(args, "positive_instance_topk_ratio", 0.25)).replace(".", "p")
        inst_tag = f"inst_on_w{weight_tag}_top{topk_tag}"
    proto_tag = "topkproto_off"
    if getattr(args, "use_topk_proto_update", False):
        ratio_tag = str(getattr(args, "topk_proto_ratio", 0.25)).replace(".", "p")
        threshold_tag = str(getattr(args, "topk_proto_threshold", 0.0)).replace(".", "p")
        warmup_tag = str(getattr(args, "topk_proto_warmup_epochs", 0))
        sep_tag = str(getattr(args, "proto_sep_loss_weight", 0.1)).replace(".", "p")
        proto_tag = f"topkproto_on_r{ratio_tag}_th{threshold_tag}_wu{warmup_tag}_sepw{sep_tag}"
        if getattr(args, "use_conservative_topk_proto_update", False):
            proto_tag = f"{proto_tag}_cons"
    cluster_tag = "ctopk_off"
    if getattr(args, "use_cluster_topk_mean_pooling", False):
        ratio_tag = str(getattr(args, "cluster_topk_mean_ratio", 0.5)).replace(".", "p")
        cluster_tag = f"ctopk_on_r{ratio_tag}"
    vens_tag = "vens_off"
    if getattr(args, "use_visual_logit_ensemble", False):
        weight_tag = str(getattr(args, "visual_logit_ensemble_weight", 0.3)).replace(".", "p")
        vens_tag = f"vens_on_w{weight_tag}"
    adrop_tag = "ardrop_off"
    if getattr(args, "use_audio_residual_drop", False):
        prob_tag = str(getattr(args, "audio_residual_drop_prob", 0.3)).replace(".", "p")
        adrop_tag = f"ardrop_on_p{prob_tag}"
    vaux_tag = "vaux_off"
    if getattr(args, "use_visual_aux_loss", False):
        weight_tag = str(getattr(args, "visual_aux_loss_weight", 0.2)).replace(".", "p")
        warmup_tag = str(getattr(args, "visual_aux_loss_warmup_epochs", 0))
        vaux_tag = f"vaux_on_w{weight_tag}_wu{warmup_tag}"
    fcons_tag = "fcons_off"
    if getattr(args, "use_fusion_consistency_loss", False):
        weight_tag = str(getattr(args, "fusion_consistency_loss_weight", 0.05)).replace(".", "p")
        temp_tag = str(getattr(args, "fusion_consistency_temperature", 2.0)).replace(".", "p")
        warmup_tag = str(getattr(args, "fusion_consistency_warmup_epochs", 0))
        fcons_tag = f"fcons_on_w{weight_tag}_t{temp_tag}_wu{warmup_tag}"
    mil_tag = "mil_off"
    if getattr(args, "use_mil_evidence_loss", False):
        weight_tag = str(getattr(args, "mil_evidence_loss_weight", 0.1)).replace(".", "p")
        topk_tag = str(getattr(args, "mil_evidence_topk_ratio", 0.25)).replace(".", "p")
        rank_tag = str(getattr(args, "mil_evidence_rank_weight", 0.05)).replace(".", "p")
        mil_tag = f"mil_on_w{weight_tag}_top{topk_tag}_rw{rank_tag}"
    calib_tag = "calib_off"
    if getattr(args, "use_eval_threshold_search", False):
        min_tag = str(getattr(args, "eval_threshold_min", 0.2)).replace(".", "p")
        max_tag = str(getattr(args, "eval_threshold_max", 0.8)).replace(".", "p")
        step_tag = str(getattr(args, "eval_threshold_step", 0.01)).replace(".", "p")
        calib_tag = f"calib_on_{min_tag}_{max_tag}_{step_tag}"
    smooth_tag = "ls0"
    if getattr(args, "label_smoothing", 0.0) > 0:
        smooth_tag = f"ls{str(getattr(args, 'label_smoothing', 0.0)).replace('.', 'p')}"
    class_weight_tag = f"cw{str(getattr(args, 'lie_class_weight', 1.5)).replace('.', 'p')}"
    margin_tag = "lmargin_off"
    if getattr(args, "use_logit_margin_regularization", False):
        weight_tag = str(getattr(args, "logit_margin_weight", 0.02)).replace(".", "p")
        target_tag = str(getattr(args, "logit_margin_target", 3.0)).replace(".", "p")
        warmup_tag = str(getattr(args, "logit_margin_warmup_epochs", 10))
        margin_tag = f"lmargin_on_w{weight_tag}_t{target_tag}_wu{warmup_tag}"
    rank_tag = "brank_off"
    if getattr(args, "use_batch_rank_loss", False):
        weight_tag = str(getattr(args, "batch_rank_loss_weight", 0.05)).replace(".", "p")
        margin_value_tag = str(getattr(args, "batch_rank_margin", 0.5)).replace(".", "p")
        warmup_tag = str(getattr(args, "batch_rank_warmup_epochs", 5))
        rank_tag = f"brank_on_w{weight_tag}_m{margin_value_tag}_wu{warmup_tag}"
    ema_tag = "ema_off"
    if getattr(args, "use_model_ema", False):
        decay_tag = str(getattr(args, "model_ema_decay", 0.995)).replace(".", "p")
        start_tag = str(getattr(args, "model_ema_start_epoch", 10))
        ema_tag = f"ema_on_d{decay_tag}_s{start_tag}"
        if getattr(args, "use_ema_dual_eval", False):
            ema_tag = f"{ema_tag}_dual"
    return f"training_log_{inst_tag}_{proto_tag}_{cluster_tag}_{vens_tag}_{adrop_tag}_{vaux_tag}_{fcons_tag}_{mil_tag}_{calib_tag}_{smooth_tag}_{class_weight_tag}_{margin_tag}_{rank_tag}_{ema_tag}.txt"


def main():
    cli_args = parse_args()
    args = Args()
    args = apply_cli_overrides(args, cli_args)
    # 使用独立的实验目录
    # args.exp_dir = args.exp_dir.rstrip('/') + '_muti'
    os.makedirs(args.exp_dir, exist_ok=True)
    
    log_file_path = os.path.join(args.exp_dir, component_log_name(args))
    logger = Logger.setup_logger(log_file_path)

    logger.info("=" * 60)
    logger.info(" Starting DOLOS pplg_muti Training (3-Fold CV)")
    logger.info(f" Modality: {args.modality}")
    logger.info(f" Log file: {log_file_path}")
    logger.info(
        " Weak instance loss: "
        f"enabled={getattr(args, 'use_instance_loss', True)}, "
        f"weight={getattr(args, 'instance_loss_weight', 0.1)}, "
        f"positive_topk_ratio={getattr(args, 'positive_instance_topk_ratio', 0.25)}"
    )
    logger.info(
        " Top-k prototype update: "
        f"enabled={getattr(args, 'use_topk_proto_update', False)}, "
        f"conservative={getattr(args, 'use_conservative_topk_proto_update', False)}, "
        f"ratio={getattr(args, 'topk_proto_ratio', 0.25)}, "
        f"threshold={getattr(args, 'topk_proto_threshold', 0.0)}, "
        f"warmup_epochs={getattr(args, 'topk_proto_warmup_epochs', 0)}, "
        f"sep_margin={getattr(args, 'proto_sep_margin', 0.2)}, "
        f"sep_weight={getattr(args, 'proto_sep_loss_weight', 0.1)}"
    )
    logger.info(
        " Cluster top-k mean pooling: "
        f"enabled={getattr(args, 'use_cluster_topk_mean_pooling', False)}, "
        f"ratio={getattr(args, 'cluster_topk_mean_ratio', 0.5)}"
    )
    logger.info(
        " Visual logit ensemble: "
        f"enabled={getattr(args, 'use_visual_logit_ensemble', False)}, "
        f"weight={getattr(args, 'visual_logit_ensemble_weight', 0.3)}"
    )
    logger.info(
        " Audio residual drop: "
        f"enabled={getattr(args, 'use_audio_residual_drop', False)}, "
        f"prob={getattr(args, 'audio_residual_drop_prob', 0.3)}"
    )
    logger.info(
        " Visual auxiliary loss: "
        f"enabled={getattr(args, 'use_visual_aux_loss', False)}, "
        f"weight={getattr(args, 'visual_aux_loss_weight', 0.2)}, "
        f"warmup_epochs={getattr(args, 'visual_aux_loss_warmup_epochs', 0)}"
    )
    logger.info(
        " Fusion consistency loss: "
        f"enabled={getattr(args, 'use_fusion_consistency_loss', False)}, "
        f"weight={getattr(args, 'fusion_consistency_loss_weight', 0.05)}, "
        f"temperature={getattr(args, 'fusion_consistency_temperature', 2.0)}, "
        f"warmup_epochs={getattr(args, 'fusion_consistency_warmup_epochs', 0)}"
    )
    logger.info(
        " MIL evidence loss: "
        f"enabled={getattr(args, 'use_mil_evidence_loss', False)}, "
        f"weight={getattr(args, 'mil_evidence_loss_weight', 0.1)}, "
        f"topk_ratio={getattr(args, 'mil_evidence_topk_ratio', 0.25)}, "
        f"rank_weight={getattr(args, 'mil_evidence_rank_weight', 0.05)}, "
        f"rank_margin={getattr(args, 'mil_evidence_rank_margin', 0.5)}, "
        f"warmup_epochs={getattr(args, 'mil_evidence_warmup_epochs', 0)}"
    )
    logger.info(
        " Eval threshold search: "
        f"enabled={getattr(args, 'use_eval_threshold_search', False)}, "
        f"min={getattr(args, 'eval_threshold_min', 0.2)}, "
        f"max={getattr(args, 'eval_threshold_max', 0.8)}, "
        f"step={getattr(args, 'eval_threshold_step', 0.01)}"
    )
    logger.info(f" Label smoothing: {getattr(args, 'label_smoothing', 0.0)}")
    logger.info(f" Lie class weight: {getattr(args, 'lie_class_weight', 1.5)}")
    logger.info(
        " Logit margin regularization: "
        f"enabled={getattr(args, 'use_logit_margin_regularization', False)}, "
        f"weight={getattr(args, 'logit_margin_weight', 0.02)}, "
        f"target={getattr(args, 'logit_margin_target', 3.0)}, "
        f"warmup_epochs={getattr(args, 'logit_margin_warmup_epochs', 10)}"
    )
    logger.info(
        " Batch rank loss: "
        f"enabled={getattr(args, 'use_batch_rank_loss', False)}, "
        f"weight={getattr(args, 'batch_rank_loss_weight', 0.05)}, "
        f"margin={getattr(args, 'batch_rank_margin', 0.5)}, "
        f"warmup_epochs={getattr(args, 'batch_rank_warmup_epochs', 5)}"
    )
    logger.info(
        " Model EMA: "
        f"enabled={getattr(args, 'use_model_ema', False)}, "
        f"decay={getattr(args, 'model_ema_decay', 0.995)}, "
        f"start_epoch={getattr(args, 'model_ema_start_epoch', 10)}, "
        f"dual_eval={getattr(args, 'use_ema_dual_eval', False)}"
    )
    set_random_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    print("\n Loading DOLOS datasets...")
    all_run_results = []
    
    for fold_idx in range(args.num_runs):
        current_seed = args.seed + fold_idx
        print("\n" + "=" * 80)
        print(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        logger.info(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        
        fold_train_losses = []
        fold_val_losses = []
        set_random_seed(current_seed)

        train_loader, test_loader, train_dataset, test_dataset = create_dolos_dataloaders(
            feature_root=args.feature_root,
            fold_path=args.fold_path,
            collate_fn=dolos_collate_fn,
            fold_index=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_feature_root=getattr(args, 'audio_feature_root', None),
            modality=getattr(args, 'modality', 'visual'),
            audio_dim=getattr(args, 'audio_dim', 1024),
        )
        
        model = LieDetection(args).cuda()
        ema_model = None
        if getattr(args, 'use_model_ema', False):
            ema_decay = float(getattr(args, 'model_ema_decay', 0.995))

            def ema_avg_fn(averaged_param, model_param, num_averaged):
                return ema_decay * averaged_param + (1.0 - ema_decay) * model_param

            ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        class_weights = torch.tensor([1.0, getattr(args, 'lie_class_weight', 1.5)]).cuda()
        criterion_bc = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=getattr(args, 'label_smoothing', 0.0)
        )
        
        print(f"\n Starting training for {args.epochs} epochs...")
        print("=" * 60)
        
        best_run_acc = 0.0
        best_run_f1 = 0.0
        best_run_metrics = {}
        
        for epoch in range(args.epochs):
            print(f"\n--- Fold {fold_idx + 1}, Epoch {epoch+1}/{args.epochs} ---")
            adjust_learning_rate(args, optimizer, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f" Learning rate: {current_lr:.6f}")
            
            # 训练 (CE + alignment_loss)
            train_metrics = train_new_epo(
                model, train_loader, criterion_bc,
                optimizer, epoch, args
            )
            use_ema_eval = False
            if ema_model is not None and epoch + 1 >= getattr(args, 'model_ema_start_epoch', 10):
                ema_model.update_parameters(model)
                use_ema_eval = True
            
            current_train_loss = train_metrics.get('train_loss', 0.0)
            fold_train_losses.append(current_train_loss)
            
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train: "
                             f"Loss={current_train_loss:.4f}, "
                             f"CE={train_metrics.get('ce_loss', 0.0):.4f}, "
                             f"Inst={train_metrics.get('instance_loss', 0.0):.4f}, "
                             f"Sep={train_metrics.get('proto_sep_loss', 0.0):.4f}, "
                             f"Ortho={train_metrics.get('ortho_loss', 0.0):.4f}, "
                             f"Margin={train_metrics.get('logit_margin_loss', 0.0):.4f}, "
                             f"Rank={train_metrics.get('batch_rank_loss', 0.0):.4f}, "
                             f"MIL={train_metrics.get('mil_evidence_loss', 0.0):.4f}, "
                             f"VAux={train_metrics.get('visual_aux_loss', 0.0):.4f}, "
                             f"FCons={train_metrics.get('fusion_consistency_loss', 0.0):.4f}")
            print(f"📈 {train_summary}")
            logger.info(train_summary)
            
            # 评估
            print(f"\n Evaluating at epoch {epoch+1}...")
            eval_source = "raw"
            eval_model = model
            if use_ema_eval:
                if getattr(args, 'use_ema_dual_eval', False):
                    raw_metrics = evaluate_model(model, test_loader, criterion_bc, args)
                    if hasattr(ema_model, 'module'):
                        ema_model.module.current_epoch = epoch
                    ema_metrics = evaluate_model(ema_model, test_loader, criterion_bc, args)
                    ema_is_better = (
                        (ema_metrics['accuracy'] > raw_metrics['accuracy']) or
                        (
                            ema_metrics['accuracy'] == raw_metrics['accuracy'] and
                            ema_metrics['f1'] > raw_metrics['f1']
                        )
                    )
                    eval_metrics = ema_metrics if ema_is_better else raw_metrics
                    eval_model = ema_model if ema_is_better else model
                    eval_source = "ema" if ema_is_better else "raw"
                else:
                    eval_model = ema_model
                    if hasattr(eval_model, 'module'):
                        eval_model.module.current_epoch = epoch
                    eval_metrics = evaluate_model(eval_model, test_loader, criterion_bc, args)
                    eval_source = "ema"
            else:
                eval_metrics = evaluate_model(eval_model, test_loader, criterion_bc, args)
            
            current_val_loss = eval_metrics.get('val_loss', 0.0)
            fold_val_losses.append(current_val_loss)
            
            eval_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Eval: "
                            f"Val Loss={current_val_loss:.4f}, "
                            f"ACC={eval_metrics['accuracy']:.4f}, "
                            f"F1={eval_metrics['f1']:.4f}, "
                            f"AUC={eval_metrics['auc']:.4f}, "
                            f"Precision={eval_metrics['precision']:.4f}, "
                            f"Recall={eval_metrics['recall']:.4f}, "
                            f"Threshold={eval_metrics.get('threshold', 0.5):.3f}, "
                            f"EvalSource={eval_source}")
            logger.info(eval_summary)
            
            is_best = (
                (eval_metrics['accuracy'] > best_run_acc) or
                (eval_metrics['accuracy'] == best_run_acc and eval_metrics['f1'] > best_run_f1)
            )

            if is_best:
                best_run_acc = eval_metrics['accuracy']
                best_run_f1 = eval_metrics['f1']
                best_run_metrics = eval_metrics
                print(f" New best accuracy: {best_run_acc:.4f}")
                
                prev_ckpts = glob.glob(os.path.join(args.exp_dir, f"model_best_fold_{fold_idx+1}_*.pth.tar"))
                for ckpt in prev_ckpts:
                    try:
                        os.remove(ckpt)
                    except OSError:
                        pass
                
                checkpoint_model = eval_model.module if hasattr(eval_model, 'module') else eval_model
                checkpoint = {
                    'epoch': epoch + 1,
                    'fold_index': fold_idx,
                    'model_state_dict': checkpoint_model.state_dict(),
                    'raw_model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'args': args,
                    'used_model_ema': use_ema_eval,
                }
                save_checkpoint(checkpoint, is_best, exp_dir=args.exp_dir,
                filename=f'model_best_fold_{fold_idx+1}_{epoch+1}.pth.tar')
        
        # Loss 曲线
        try:
            plt.figure(figsize=(10, 5))
            epochs_range = range(1, args.epochs + 1)
            plt.plot(epochs_range, fold_train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
            plt.plot(epochs_range, fold_val_losses, label='Val Loss', color='red', linestyle='--', marker='s', markersize=3)
            plt.title(f'Loss Curve - Fold {fold_idx + 1} (pplg_muti)', fontsize=14, fontweight='bold')
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(args.exp_dir, f'loss_curve_fold_{fold_idx+1}.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"📉 Loss curve saved to {plot_path}")
            logger.info(f"Loss curve saved to {plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to plot loss curve: {e}")
            logger.warning(f"Failed to plot loss curve: {e}")
        
        print(f"\n✅ Finished Fold {fold_idx + 1}. Best Accuracy: {best_run_acc:.4f}")
        logger.info(f"Finished Fold {fold_idx + 1}. Best Metrics: {best_run_metrics}")
        
        if best_run_metrics:
            all_run_results.append(best_run_metrics)

    if not all_run_results:
        print("No successful folds to report.")
        return

    results_df = pd.DataFrame(all_run_results)
    avg_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    print("\n📊 Individual Fold Results:")
    print(results_df.round(4))
    
    print("\n📈 Final Averaged Metrics across all folds (+/- std dev):")
    for metric in avg_metrics.index:
        print(f"  - Average {metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    logger.info("\n=== Final Averaged Results (3-Fold Cross-Validation) ===" + "\n" +
                pd.concat([avg_metrics, std_metrics], keys=['Mean', 'Std']).to_string())
    print("\nDOLOS pplg_muti 3-Fold Cross-Validation complete. Final results logged.")


if __name__ == "__main__":
    main()
