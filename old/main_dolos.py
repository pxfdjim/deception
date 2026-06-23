
import os, glob
import matplotlib.pyplot as plt  # 引入绘图库
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from utils.eval import evaluate_model
from models.pplg import LieDetection
from utils.ins_loss import  adjust_learning_rate
from utils.logger import Logger
from utils.utils import set_random_seed,save_checkpoint
from datasets.dataloaderFactory import create_dolos_dataloaders  # 使用DOLOS数据加载器
from datasets.dolos import dolos_collate_fn  # 使用DOLOS的collate函数
from configs.dolos import Args  # 使用config.py中的Args
from utils.train import train_epoch

def main():
    args = Args()
    
    os.makedirs(args.exp_dir, exist_ok=True)
    log_file_path = os.path.join(args.exp_dir, 'training_log.txt')
    logger = Logger.setup_logger(log_file_path)

    logger.info("="*60)
    logger.info(" Starting PPLG-based Lie Detection Training")
    set_random_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    print("\n Loading DOLOS datasets...")
    all_run_results = []
    # DOLOS使用3折交叉验证，每个fold作为一次run
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

        # 使用DOLOS数据加载器
        train_loader, test_loader, train_dataset, test_dataset = create_dolos_dataloaders(
            feature_root=args.feature_root,
            fold_path=args.fold_path,
            collate_fn=dolos_collate_fn,  # 使用DOLOS的collate函数
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
            optimizer = optim.Adam(
                model.parameters(), lr=1e-4, weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )
        class_weights = torch.tensor([1.0, 1.5]).cuda()
        criterion_bc = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\n Starting training for {args.epochs} epochs...")
        print("=" * 60)
        # best_acc  = 0.0
        best_run_acc = 0.0
        best_run_f1 = 0.0
        best_run_metrics = {}
        for epoch in range(args.epochs):
            
            print(f"\n--- Fold {fold_idx + 1}, Epoch {epoch+1}/{args.epochs} ---")
            adjust_learning_rate(args, optimizer, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f" Learning rate: {current_lr:.6f}")
            
            # 训练一个epoch
            train_metrics = train_epoch(
                model, train_loader, criterion_bc, 
                optimizer, epoch, args
            )
            
            # 记录训练 Loss
            current_train_loss = train_metrics.get('train_loss', 0.0)
            fold_train_losses.append(current_train_loss)
            
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train Summary: "
                             f"Train Loss={current_train_loss:.4f}")
            print(f"📈 {train_summary}")
            logger.info(train_summary)
            
            
            # 评估模型 (每个epoch都验证，以便画出平滑的曲线)
            print(f"\n Evaluating at epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, test_loader, criterion_bc, args)
            
            # 记录验证 Loss
            current_val_loss = eval_metrics.get('val_loss', 0.0)
            fold_val_losses.append(current_val_loss)
            
            eval_summary = (f"Evaluation Results at Fold {fold_idx+1} Epoch {epoch+1} Eval: "
                                f"Val Loss={current_val_loss:.4f}, "
                                f"ACC={eval_metrics['accuracy']:.4f}, "
                                f"F1={eval_metrics['f1']:.4f}, "
                                f"AUC={eval_metrics['auc']:.4f}, "
                                f"Precision={eval_metrics['precision']:.4f}, "
                                f"Recall={eval_metrics['recall']:.4f}")
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
                        print(f"🗑️ Removed old checkpoint: {ckpt}")
                    except OSError as e:
                        print(f"⚠️ Cannot remove {ckpt}: {e}")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'fold_index': fold_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics, 'eval_metrics': eval_metrics, 'args': args
                }, is_best, exp_dir=args.exp_dir,                       filename=f'model_best_fold_{fold_idx+1}_{epoch+1}.pth.tar')
        
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
        
        print(f"\n✅ Finished Fold {fold_idx + 1}. Best Accuracy achieved: {best_run_acc:.4f}")
        logger.info(f"Finished Fold {fold_idx + 1}. Best Metrics: {best_run_metrics}")    
      
        if best_run_metrics: # Ensure we don't append an empty dict if training fails
             all_run_results.append(best_run_metrics)
    if not all_run_results:
        print("No successful folds to report.")
        return

    # Use pandas to calculate and display the final summary
    results_df = pd.DataFrame(all_run_results)
    avg_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    print("\n📊 Individual Fold Results:")
    print(results_df.round(4))
    
    print("\n📈 Final Averaged Metrics across all folds (+/- std dev):")
    for metric in avg_metrics.index:
        print(f"  - Average {metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    logger.info("\n=== Final Averaged Results (3-Fold Cross-Validation) ===" + "\n" + pd.concat([avg_metrics, std_metrics], keys=['Mean', 'Std']).to_string())
    print("\nDOLOS 3-Fold Cross-Validation training and evaluation complete. Final results logged.")    



if __name__ == "__main__":
    main()