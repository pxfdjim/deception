import os, glob
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
from models.pplg import LieDetection
from utils.ins_loss import adjust_learning_rate
from utils.logger import Logger
from utils.utils import set_random_seed, save_checkpoint
from datasets.newData import (
    get_subject_loader,
    compute_topk_accuracy
)
from configs.newData import Args
from utils.train import train_newdata_ranking_epoch


def main():
    args = Args()
    
    os.makedirs(args.exp_dir, exist_ok=True)
    log_file_path = os.path.join(args.exp_dir, 'training_log.txt')
    logger = Logger.setup_logger(log_file_path)

    logger.info("="*60)
    logger.info(" Starting NumberGuess Lie Detection Training (5-Fold Cross-Validation)")
    set_random_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    print("\n Loading NumberGuess datasets...")
    all_run_results = []
    
    # NumberGuess使用5折交叉验证
    for fold_idx in range(args.num_runs):
        current_seed = args.seed + fold_idx
        print("\n" + "="*80)
        print(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        logger.info(f"=============== Starting Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ===============")
        
        # 初始化用于记录当前 Fold 的 Loss 列表
        fold_train_losses = []
        fold_val_losses = []
        
        # Set the seed for all libraries for this fold
        set_random_seed(current_seed)

        # 创建被试级别的 DataLoader（训练 + 验证）
        train_subject_loader = get_subject_loader(
            fold=fold_idx + 1,
            batch_size=getattr(args, 'subject_batch_size', 4),
            num_workers=args.num_workers,
            feature_dir=args.feature_root,
            split="train",
            audio_feature_dir=getattr(args, 'audio_feature_root', None),
            modality=getattr(args, 'modality', 'visual'),
            audio_dim=getattr(args, 'audio_dim', 1024),
        )
        
        model = LieDetection(args).cuda()
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 检查训练数据的类别分布（从 SubjectDataset 统计）
        print("\n[数据分布检查]")
        label_counts = {0: 0, 1: 0}
        for _, _, batch_labels, _, _ in train_subject_loader:
            valid = batch_labels[batch_labels != -1]
            for label in valid:
                label_counts[label.item()] += 1
        total = sum(label_counts.values())
        ratio = label_counts[1] / label_counts[0] if label_counts[0] > 0 else 1.0
        print(f"训练集标签分布: 真实(0)={label_counts[0]} ({label_counts[0]/total*100:.1f}%), "
              f"谎言(1)={label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
        print(f"类别比例: 1:{ratio:.2f}")
        
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )
        
        # 根据实际分布动态计算类别权重
        weight_1 = label_counts[0] / label_counts[1] if label_counts[1] > 0 else 2.0
        weight_1 = min(weight_1, 2.5)
        weight_1 = max(weight_1, 1.5)
        class_weights = torch.tensor([1.0, weight_1]).cuda()
        print(f"使用类别权重: [1.0, {weight_1:.2f}]")
        
        criterion_bc = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\n Starting training for {args.epochs} epochs...")
        print("=" * 60)
        
        # 元组级联比较: (Top-1, Top-2, Top-3, Top-5)，Top-1优先，相同看Top-2，以此类推
        best_topk_tuple = (-1.0, -1.0, -1.0, -1.0)
        best_run_metrics = {}
        
        for epoch in range(args.epochs):
            print(f"\n--- Fold {fold_idx + 1}, Epoch {epoch+1}/{args.epochs} ---")
            adjust_learning_rate(args, optimizer, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f" Learning rate: {current_lr:.6f}")
            
            # 训练一个epoch（被试内排序训练）
            train_metrics = train_newdata_ranking_epoch(
                model, train_subject_loader, criterion_bc, 
                optimizer, epoch, args
            )
            
            # 记录训练 Loss
            current_train_loss = train_metrics.get('train_loss', 0.0)
            ce_loss = train_metrics.get('ce_loss', 0.0)
            rank_loss = train_metrics.get('rank_loss', 0.0)
            fold_train_losses.append(current_train_loss)
            
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train Summary: "
                             f"Loss={current_train_loss:.4f}, CE={ce_loss:.4f}, Rank={rank_loss:.4f}")
            print(f"📈 {train_summary}")
            logger.info(train_summary)
            
            # Top-K 评估（每个epoch）
            print(f"\n Computing Top-K accuracy at epoch {epoch+1}...")
            topk_metrics = {}
            
            # 创建被试级别的 DataLoader（含音频特征）
            subject_loader = get_subject_loader(
                fold=fold_idx + 1,  # fold从1开始
                batch_size=8,
                num_workers=args.num_workers,
                feature_dir=args.feature_root,
                split="val",
                audio_feature_dir=getattr(args, 'audio_feature_root', None),
                modality=getattr(args, 'modality', 'visual'),
                audio_dim=getattr(args, 'audio_dim', 1024),
            )
            
            # 创建模型适配器，将 padded tensor 转为 features_list 送入模型
            class ModelAdapter(nn.Module):
                def __init__(self, model, modality='visual'):
                    super().__init__()
                    self.model = model
                    self.modality = modality
                
                def forward(self, v_feats, a_feats):
                    # v_feats: (B, max_seg_v, v_dim), a_feats: (B, max_seg_a, a_dim)
                    visual_list = [v_feats[i] for i in range(v_feats.shape[0])]
                    audio_list  = [a_feats[i] for i in range(a_feats.shape[0])]
                    
                    if self.modality == 'both':
                        outputs = self.model(visual_list, audio_list, bag_labels=None)
                    elif self.modality == 'audio':
                        outputs = self.model(None, audio_list, bag_labels=None)
                    else:
                        outputs = self.model(visual_list, None, bag_labels=None)
                    return outputs['logits']  # (B, 2)
            
            model_adapter = ModelAdapter(model, modality=getattr(args, 'modality', 'visual'))
            topk_results = compute_topk_accuracy(model_adapter, subject_loader, torch.device('cuda'), ks=(1, 2, 3, 5))
            topk_metrics = topk_results
            
            print(f"\n📊 Top-K Accuracy Results:")
            print(f"   Top-1: {topk_results[1]:.4f}")
            print(f"   Top-2: {topk_results[2]:.4f}")
            print(f"   Top-3: {topk_results[3]:.4f}")
            print(f"   Top-5: {topk_results[5]:.4f}")
            
            # 记录到日志
            eval_summary = (f"Fold {fold_idx+1} Epoch {epoch+1}: "
                           f"Train Loss={current_train_loss:.4f}, "
                           f"Top-1={topk_results[1]:.4f}, "
                           f"Top-2={topk_results[2]:.4f}, "
                           f"Top-3={topk_results[3]:.4f}, "
                           f"Top-5={topk_results[5]:.4f}")
            logger.info(eval_summary)
            
            # 用于保存验证loss的占位符（用于绘图）
            current_val_loss = 1.0 - topk_results[2]  # 使用 Top-2 作为主要指标
            fold_val_losses.append(current_val_loss)
            
            # === 最佳模型: 元组级联比较 Top-1 > Top-2 > Top-3 > Top-5 ===
            current_tuple = (topk_results[1], topk_results[2], topk_results[3], topk_results[5])
            is_best = current_tuple > best_topk_tuple

            if is_best:
                best_topk_tuple = current_tuple
                best_run_metrics = topk_metrics
                print(f"✨ New best: Top-1={current_tuple[0]:.4f}, Top-2={current_tuple[1]:.4f}, "
                      f"Top-3={current_tuple[2]:.4f}, Top-5={current_tuple[3]:.4f}")
                
                # 删除旧的 checkpoint
                prev_ckpts = glob.glob(os.path.join(args.exp_dir, f"model_best_fold_{fold_idx+1}_*.pth.tar"))
                for ckpt in prev_ckpts:
                    try:
                        os.remove(ckpt)
                    except OSError:
                        pass
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'fold_index': fold_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': {'train_loss': current_train_loss},
                    'topk_metrics': topk_metrics,
                    'args': args
                }, True, exp_dir=args.exp_dir,
                filename=f'model_best_fold_{fold_idx+1}_ep{epoch+1}.pth.tar')
        
        # Fold 结束时，绘制并保存 Loss 曲线
        try:
            plt.figure(figsize=(10, 5))
            epochs_range = range(1, args.epochs + 1)
            plt.plot(epochs_range, fold_train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
            plt.plot(epochs_range, fold_val_losses, label='Val Loss', color='red', linestyle='--', marker='s', markersize=3)
            plt.title(f'Loss Curve - Fold {fold_idx + 1}', fontsize=14, fontweight='bold')
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
        
        fold_summary = (f"\n✅ Finished Fold {fold_idx + 1}. "
                       f"Best Top-1: {best_topk_tuple[0]:.4f}, Top-2: {best_topk_tuple[1]:.4f}, "
                       f"Top-3: {best_topk_tuple[2]:.4f}, Top-5: {best_topk_tuple[3]:.4f}")
        print(fold_summary)
        logger.info(f"Finished Fold {fold_idx + 1}. Best: {best_topk_tuple}")    
      
        if best_run_metrics:
            all_run_results.append(best_run_metrics)
    
    if not all_run_results:
        print("No successful folds to report.")
        return

    # 使用pandas计算并显示最终汇总
    results_df = pd.DataFrame(all_run_results)
    avg_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    print("\n📊 Individual Fold Results:")
    print(results_df.round(4))
    
    print("\n📈 Final Averaged Top-K Metrics across all folds (+/- std dev):")
    for metric in avg_metrics.index:
        metric_name = metric.replace('_', '-').upper() if isinstance(metric, str) else str(metric)
        print(f"  - {metric_name}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    logger.info("\n=== Final Averaged Results (5-Fold Cross-Validation) ===" + "\n" + 
                pd.concat([avg_metrics, std_metrics], keys=['Mean', 'Std']).to_string())
    print("\nNumberGuess 5-Fold Cross-Validation training and evaluation complete. Final results logged.")


if __name__ == "__main__":
    main()
