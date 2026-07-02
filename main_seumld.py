"""
SEUMLD 数据集 - pplg_muti 非对称多模态模型训练脚本
视觉: MIL 原型引导层次化聚合
音频: 全局 MLP 编码
融合: 门控融合 + 正交正则化损失
5折交叉验证
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
import pandas as pd
import numpy as np
from utils.eval import evaluate_model
from models.final import LieDetection
from utils.ins_loss import adjust_learning_rate
from utils.logger import Logger
from utils.utils import set_random_seed, save_checkpoint
from datasets.dataloaderFactory import create_seumld_dataloaders
from datasets.seumld import seumld_multimodal_collate_fn, finegrained_collate_fn
from configs.seumld import Args
from utils.train import train_seumld_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="SEUMLD component ablation training")
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
    parser.add_argument("--use-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_true", default=None)
    parser.add_argument("--no-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_false")
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=None)
    parser.add_argument("--exp-suffix", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--best-epoch-objective", choices=["acc_f1", "acc_tolerant_f1"], default=None)
    parser.add_argument("--best-epoch-acc-tolerance", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-epochs", type=int, default=None)
    parser.add_argument("--disable-tqdm", dest="disable_tqdm", action="store_true", default=None)
    parser.add_argument("--enable-tqdm", dest="disable_tqdm", action="store_false")
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
        "use_cluster_topk_mean_pooling",
        "cluster_topk_mean_ratio",
        "epochs",
        "num_runs",
        "batch_size",
        "lr",
        "num_workers",
        "seed",
        "eval_every",
        "best_epoch_objective",
        "best_epoch_acc_tolerance",
        "early_stop_patience",
        "early_stop_min_epochs",
        "disable_tqdm",
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
        proto_tag = f"topkproto_on_r{ratio_tag}_th{threshold_tag}_wu{warmup_tag}"
        if getattr(args, "use_conservative_topk_proto_update", False):
            proto_tag = f"{proto_tag}_cons"
    cluster_tag = "ctopk_off"
    if getattr(args, "use_cluster_topk_mean_pooling", False):
        ratio_tag = str(getattr(args, "cluster_topk_mean_ratio", 0.5)).replace(".", "p")
        cluster_tag = f"ctopk_on_r{ratio_tag}"
    eval_tag = f"eval{int(getattr(args, 'eval_every', 1))}"
    best_tag = str(getattr(args, "best_epoch_objective", "acc_f1"))
    best_tol_tag = str(getattr(args, "best_epoch_acc_tolerance", 0.0)).replace(".", "p")
    lr_tag = f"lr{str(getattr(args, 'lr', 0.05)).replace('.', 'p')}"
    bs_tag = f"bs{getattr(args, 'batch_size', 8)}"
    patience = int(getattr(args, "early_stop_patience", 0) or 0)
    min_epochs = int(getattr(args, "early_stop_min_epochs", 0) or 0)
    early_stop_tag = "es_off" if patience <= 0 else f"es{patience}_min{min_epochs}"
    tqdm_tag = "tqdm_off" if getattr(args, "disable_tqdm", False) else "tqdm_on"
    return f"training_log_{inst_tag}_{proto_tag}_{cluster_tag}_{eval_tag}_best{best_tag}_tol{best_tol_tag}_{early_stop_tag}_{lr_tag}_{bs_tag}_{tqdm_tag}.txt"


def main():
    cli_args = parse_args()
    args = Args()
    args = apply_cli_overrides(args, cli_args)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    log_file_path = os.path.join(args.exp_dir, component_log_name(args))
    logger = Logger.setup_logger(log_file_path)

    logger.info("=" * 60)
    logger.info(" Starting SEUMLD pplg_muti Training (5-Fold CV)")
    logger.info(f" Modality: {args.modality}")
    logger.info(f" Visual feature: {args.feature_root}")
    logger.info(f" Audio feature: {args.audio_feature_root}")
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
        f"warmup_epochs={getattr(args, 'topk_proto_warmup_epochs', 0)}"
    )
    logger.info(
        " Cluster top-k mean pooling: "
        f"enabled={getattr(args, 'use_cluster_topk_mean_pooling', False)}, "
        f"ratio={getattr(args, 'cluster_topk_mean_ratio', 0.5)}"
    )
    logger.info(
        " Best epoch selection: "
        f"objective={getattr(args, 'best_epoch_objective', 'acc_f1')}, "
        f"acc_tolerance={getattr(args, 'best_epoch_acc_tolerance', 0.0)}"
    )
    logger.info(
        " Runtime overrides: "
        f"epochs={getattr(args, 'epochs', 100)}, "
        f"batch_size={getattr(args, 'batch_size', 8)}, "
        f"lr={getattr(args, 'lr', 0.05)}, "
        f"num_workers={getattr(args, 'num_workers', 1)}, "
        f"eval_every={getattr(args, 'eval_every', 1)}, "
        f"early_stop_patience={getattr(args, 'early_stop_patience', 0)}, "
        f"early_stop_min_epochs={getattr(args, 'early_stop_min_epochs', 0)}, "
        f"disable_tqdm={getattr(args, 'disable_tqdm', False)}, "
        f"seed={getattr(args, 'seed', 42)}"
    )
    set_random_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    print("\n Loading SEUMLD datasets...")
    all_run_results = []
    
    for fold_idx in range(args.num_runs):
        current_seed = args.seed + fold_idx
        print("\n" + "=" * 80)
        print(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        logger.info(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        
        fold_train_losses = []
        fold_val_losses = []
        fold_val_epochs = []
        set_random_seed(current_seed)

        # 根据模态选择collate_fn
        if args.modality == 'visual':
            collate_fn = finegrained_collate_fn
        else:
            collate_fn = seumld_multimodal_collate_fn
        
        # 创建数据加载器
        train_loader, test_loader, train_dataset, test_dataset = create_seumld_dataloaders(
            feature_root=args.feature_root,
            label_path=args.label_path,
            fold_path=args.fold_path,
            collate_fn=collate_fn,
            fold_index=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_feature_root=getattr(args, 'audio_feature_root', None),
            modality=getattr(args, 'modality', 'visual'),
            audio_dim=getattr(args, 'audio_dim', 1024),
        )
        
        # 检查训练数据的类别分布
        print("\n[数据分布检查]")
        label_counts = {0: 0, 1: 0}
        for _, _, batch_labels, _ in train_loader:
            for label in batch_labels:
                label_counts[label.item()] += 1
        total = sum(label_counts.values())
        if total > 0:
            ratio = label_counts[1] / label_counts[0] if label_counts[0] > 0 else 1.0
            print(f"训练集标签分布: 真实(0)={label_counts[0]} ({label_counts[0]/total*100:.1f}%), "
                  f"谎言(1)={label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
            print(f"类别比例: 1:{ratio:.2f}")
        
        model = LieDetection(args).cuda()
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 配置优化器
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 动态类别权重
        if label_counts[1] > 0 and label_counts[0] > 0:
            weight_1 = label_counts[0] / label_counts[1]
            weight_1 = min(max(weight_1, 1.0), 3.0)  # 限制在 [1.0, 3.0]
        else:
            weight_1 = 1.5
        class_weights = torch.tensor([1.0, weight_1]).cuda()
        print(f"使用类别权重: [1.0, {weight_1:.2f}]")
        
        criterion_bc = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\n Starting training for {args.epochs} epochs...")
        print("=" * 60)
        
        best_run_acc = 0.0
        best_run_f1 = 0.0
        best_run_metrics = {}
        stale_evals = 0
        
        for epoch in range(args.epochs):
            print(f"\n--- Fold {fold_idx + 1}, Epoch {epoch+1}/{args.epochs} ---")
            adjust_learning_rate(args, optimizer, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f" Learning rate: {current_lr:.6f}")
            
            # 训练
            train_metrics = train_seumld_epoch(
                model, train_loader, criterion_bc,
                optimizer, epoch, args
            )
            
            current_train_loss = train_metrics.get('train_loss', 0.0)
            fold_train_losses.append(current_train_loss)
            
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train: "
                             f"Loss={current_train_loss:.4f}, "
                             f"CE={train_metrics.get('ce_loss', 0.0):.4f}, "
                             f"Inst={train_metrics.get('instance_loss', 0.0):.4f}, "
                             f"Ortho={train_metrics.get('ortho_loss', 0.0):.4f}")
            print(f"📈 {train_summary}")
            logger.info(train_summary)

            should_eval = (
                (epoch + 1) == args.epochs or
                (epoch + 1) % max(1, int(getattr(args, "eval_every", 1))) == 0
            )
            if not should_eval:
                continue
            
            # 评估
            print(f"\n Evaluating at epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, test_loader, criterion_bc, args)
            
            current_val_loss = eval_metrics.get('val_loss', 0.0)
            fold_val_losses.append(current_val_loss)
            fold_val_epochs.append(epoch + 1)
            
            eval_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Eval: "
                            f"Val Loss={current_val_loss:.4f}, "
                            f"ACC={eval_metrics['accuracy']:.4f}, "
                            f"F1={eval_metrics['f1']:.4f}, "
                            f"AUC={eval_metrics['auc']:.4f}, "
                            f"Precision={eval_metrics['precision']:.4f}, "
                            f"Recall={eval_metrics['recall']:.4f}, "
                            f"Threshold={eval_metrics.get('threshold', 0.5):.3f}")
            logger.info(eval_summary)
            
            # 最佳模型判断
            best_objective = getattr(args, 'best_epoch_objective', 'acc_f1')
            if best_objective == 'acc_tolerant_f1':
                best_tol = float(getattr(args, 'best_epoch_acc_tolerance', 0.0))
                is_best = (
                    (eval_metrics['accuracy'] > best_run_acc + best_tol) or
                    (
                        eval_metrics['accuracy'] >= best_run_acc - best_tol and
                        eval_metrics['f1'] > best_run_f1
                    )
                )
            else:
                is_best = (
                    (eval_metrics['accuracy'] > best_run_acc) or
                    (eval_metrics['accuracy'] == best_run_acc and eval_metrics['f1'] > best_run_f1)
                )

            if is_best:
                best_run_acc = eval_metrics['accuracy']
                best_run_f1 = eval_metrics['f1']
                best_run_metrics = eval_metrics
                stale_evals = 0
                print(f"✨ New best accuracy: {best_run_acc:.4f}")
                
                # 删除旧的checkpoint
                prev_ckpts = glob.glob(os.path.join(args.exp_dir, f"model_best_fold_{fold_idx+1}_*.pth.tar"))
                for ckpt in prev_ckpts:
                    try:
                        os.remove(ckpt)
                    except OSError:
                        pass
                
                # 保存新的checkpoint
                save_checkpoint({
                    'epoch': epoch + 1,
                    'fold_index': fold_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'args': args
                }, is_best, exp_dir=args.exp_dir,
                filename=f'model_best_fold_{fold_idx+1}_{epoch+1}.pth.tar')
            else:
                stale_evals += 1

            patience = int(getattr(args, "early_stop_patience", 0) or 0)
            min_epochs = int(getattr(args, "early_stop_min_epochs", 0) or 0)
            if patience > 0 and (epoch + 1) >= min_epochs and stale_evals >= patience:
                stop_msg = (
                    f"Fold {fold_idx+1} early stopped at epoch {epoch+1}: "
                    f"no best update for {stale_evals} evals "
                    f"(patience={patience}, min_epochs={min_epochs})."
                )
                print(f"⏹ {stop_msg}")
                logger.info(stop_msg)
                break
        
        # Loss 曲线
        try:
            plt.figure(figsize=(10, 5))
            epochs_range = range(1, len(fold_train_losses) + 1)
            plt.plot(epochs_range, fold_train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
            plt.plot(fold_val_epochs, fold_val_losses, label='Val Loss', color='red', linestyle='--', marker='s', markersize=3)
            plt.title(f'Loss Curve - Fold {fold_idx + 1} (SEUMLD)', fontsize=14, fontweight='bold')
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

    # 计算平均结果
    results_df = pd.DataFrame(all_run_results)
    avg_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    print("\n📊 Individual Fold Results:")
    print(results_df.round(4))
    
    print("\n📈 Final Averaged Metrics across all folds (+/- std dev):")
    for metric in avg_metrics.index:
        print(f"  - Average {metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    logger.info("\n=== Final Averaged Results (5-Fold Cross-Validation) ===" + "\n" +
                pd.concat([avg_metrics, std_metrics], keys=['Mean', 'Std']).to_string())
    print("\nSEUMLD pplg_muti 5-Fold Cross-Validation complete. Final results logged.")


if __name__ == "__main__":
    main()
