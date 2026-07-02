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
import pandas as pd
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
    parser.add_argument("--use-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_true", default=None)
    parser.add_argument("--no-cluster-topk-mean-pooling", dest="use_cluster_topk_mean_pooling", action="store_false")
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=None)
    parser.add_argument("--exp-suffix", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--lie-class-weight", type=float, default=None)
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
        "label_smoothing",
        "lie_class_weight",
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
        proto_tag = f"topkproto_on_r{ratio_tag}_th{threshold_tag}_wu{warmup_tag}"
        if getattr(args, "use_conservative_topk_proto_update", False):
            proto_tag = f"{proto_tag}_cons"
    cluster_tag = "ctopk_off"
    if getattr(args, "use_cluster_topk_mean_pooling", False):
        ratio_tag = str(getattr(args, "cluster_topk_mean_ratio", 0.5)).replace(".", "p")
        cluster_tag = f"ctopk_on_r{ratio_tag}"
    smooth_tag = "ls0"
    if getattr(args, "label_smoothing", 0.0) > 0:
        smooth_tag = f"ls{str(getattr(args, 'label_smoothing', 0.0)).replace('.', 'p')}"
    class_weight_tag = f"cw{str(getattr(args, 'lie_class_weight', 1.5)).replace('.', 'p')}"
    return f"training_log_{inst_tag}_{proto_tag}_{cluster_tag}_{smooth_tag}_{class_weight_tag}.txt"


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
        f"warmup_epochs={getattr(args, 'topk_proto_warmup_epochs', 0)}"
    )
    logger.info(
        " Cluster top-k mean pooling: "
        f"enabled={getattr(args, 'use_cluster_topk_mean_pooling', False)}, "
        f"ratio={getattr(args, 'cluster_topk_mean_ratio', 0.5)}"
    )
    logger.info(f" Label smoothing: {getattr(args, 'label_smoothing', 0.0)}")
    logger.info(f" Lie class weight: {getattr(args, 'lie_class_weight', 1.5)}")
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
            
            current_train_loss = train_metrics.get('train_loss', 0.0)
            fold_train_losses.append(current_train_loss)
            
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train: "
                             f"Loss={current_train_loss:.4f}, "
                             f"CE={train_metrics.get('ce_loss', 0.0):.4f}, "
                             f"Inst={train_metrics.get('instance_loss', 0.0):.4f}, "
                             f"Ortho={train_metrics.get('ortho_loss', 0.0):.4f}")
            print(f"📈 {train_summary}")
            logger.info(train_summary)
            
            # 评估
            print(f"\n Evaluating at epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, test_loader, criterion_bc, args)
            
            current_val_loss = eval_metrics.get('val_loss', 0.0)
            fold_val_losses.append(current_val_loss)
            
            eval_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Eval: "
                            f"Val Loss={current_val_loss:.4f}, "
                            f"ACC={eval_metrics['accuracy']:.4f}, "
                            f"F1={eval_metrics['f1']:.4f}, "
                            f"AUC={eval_metrics['auc']:.4f}, "
                            f"Precision={eval_metrics['precision']:.4f}, "
                            f"Recall={eval_metrics['recall']:.4f}, "
                            f"Threshold={eval_metrics.get('threshold', 0.5):.3f}")
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
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'fold_index': fold_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'args': args,
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
