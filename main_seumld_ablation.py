"""
SEUMLD 数据集 - 多模态融合消融实验脚本
=====================================

运行所有消融配置，输出对比表格

用法:
    python main_seumld_ablation.py              # 运行所有消融模式
    python main_seumld_ablation.py --mode full  # 运行特定模式
"""
import os
import sys
import glob
import argparse
import copy
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime

from utils.eval import evaluate_model
from models.pplg_ablation import LieDetectionAblation, get_ablation_modes, print_ablation_table
from utils.ins_loss import adjust_learning_rate
from utils.logger import Logger
from utils.utils import set_random_seed, save_checkpoint
from datasets.dataloaderFactory import create_seumld_dataloaders
from datasets.seumld import seumld_multimodal_collate_fn, finegrained_collate_fn
from configs.seumld import Args
from utils.train import train_seumld_epoch


def run_single_ablation(ablation_mode, args, logger, device):
    """运行单个消融配置的 5 折交叉验证"""
    
    # 设置消融模式
    args.fusion_mode = ablation_mode
    
    # 根据消融模式调整 modality
    if ablation_mode == 'only_visual':
        args.modality = 'visual'
    elif ablation_mode == 'only_audio':
        args.modality = 'audio'
    else:
        args.modality = 'both'
    
    print(f"\n{'='*80}")
    print(f"  Ablation Mode: {ablation_mode}")
    print(f"  Modality: {args.modality}")
    print(f"{'='*80}")
    logger.info(f"\n{'='*80}")
    logger.info(f"  Ablation Mode: {ablation_mode}")
    logger.info(f"  Modality: {args.modality}")
    logger.info(f"{'='*80}")
    
    all_fold_results = []
    fold_details = []  # 保存每个fold的详细指标
    
    for fold_idx in range(args.num_runs):
        current_seed = args.seed + fold_idx
        print(f"\n--- Fold {fold_idx + 1}/{args.num_runs} (Seed: {current_seed}) ---")
        
        set_random_seed(current_seed)
        
        # 选择 collate_fn
        if args.modality == 'visual':
            collate_fn = finegrained_collate_fn
        else:
            collate_fn = seumld_multimodal_collate_fn
        
        # 创建数据加载器
        train_loader, test_loader, _, _ = create_seumld_dataloaders(
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
        
        # 统计类别分布
        label_counts = {0: 0, 1: 0}
        for _, _, batch_labels, _ in train_loader:
            for label in batch_labels:
                label_counts[label.item()] += 1
        
        # 创建模型
        model = LieDetectionAblation(args).to(device)
        
        # 配置优化器
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 类别权重
        if label_counts[1] > 0 and label_counts[0] > 0:
            weight_1 = min(max(label_counts[0] / label_counts[1], 1.0), 3.0)
        else:
            weight_1 = 1.5
        class_weights = torch.tensor([1.0, weight_1]).to(device)
        criterion_bc = nn.CrossEntropyLoss(weight=class_weights)
        
        best_fold_acc = 0.0
        best_fold_f1 = 0.0
        best_fold_metrics = {}
        
        for epoch in range(args.epochs):
            adjust_learning_rate(args, optimizer, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\n--- Fold {fold_idx + 1}/{args.num_runs}, Epoch {epoch+1}/{args.epochs} ---")
            print(f"  Learning rate: {current_lr:.6f}")
            logger.info(f"--- Fold {fold_idx + 1}/{args.num_runs}, Epoch {epoch+1}/{args.epochs} ---")
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # 训练
            train_metrics = train_seumld_epoch(
                model, train_loader, criterion_bc,
                optimizer, epoch, args
            )
            
            # 打印训练指标
            train_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Train: "
                             f"Loss={train_metrics.get('train_loss', 0.0):.4f}, "
                             f"CE={train_metrics.get('ce_loss', 0.0):.4f}, "
                             f"Ortho={train_metrics.get('ortho_loss', 0.0):.4f}")
            print(f"  📈 {train_summary}")
            logger.info(f"Train: {train_summary}")
            
            # 评估
            print(f"  Evaluating at epoch {epoch+1}...")
            logger.info(f"Evaluating at epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, test_loader, criterion_bc, args)
            
            # 打印评估指标
            eval_summary = (f"Fold {fold_idx+1} Epoch {epoch+1} Eval: "
                            f"Val Loss={eval_metrics.get('val_loss', 0.0):.4f}, "
                            f"ACC={eval_metrics['accuracy']:.4f}, "
                            f"F1={eval_metrics['f1']:.4f}, "
                            f"AUC={eval_metrics['auc']:.4f}")
            print(f"  📊 {eval_summary}")
            logger.info(f"Eval: {eval_summary}")
            
            # 记录最佳
            is_best = (eval_metrics['accuracy'] > best_fold_acc) or \
                      (eval_metrics['accuracy'] == best_fold_acc and eval_metrics['f1'] > best_fold_f1)
            if is_best:
                best_fold_acc = eval_metrics['accuracy']
                best_fold_f1 = eval_metrics['f1']
                best_fold_metrics = eval_metrics.copy()
                print(f"  ✨ New best accuracy: {best_fold_acc:.4f}")
                logger.info(f"New best accuracy: {best_fold_acc:.4f}")
        
        print(f"\n  Fold {fold_idx + 1} Best: ACC={best_fold_acc:.4f}, F1={best_fold_f1:.4f}")
        logger.info(f"Fold {fold_idx + 1} Best: ACC={best_fold_acc:.4f}, F1={best_fold_f1:.4f}")
        
        if best_fold_metrics:
            all_fold_results.append(best_fold_metrics)
            # 记录每个fold的详细指标
            fold_detail = {
                'mode': ablation_mode,
                'fold': fold_idx + 1,
                'accuracy': best_fold_metrics.get('accuracy', 0),
                'f1': best_fold_metrics.get('f1', 0),
                'auc': best_fold_metrics.get('auc', 0),
                'precision': best_fold_metrics.get('precision', 0),
                'recall': best_fold_metrics.get('recall', 0),
            }
            fold_details.append(fold_detail)
    
    # 计算平均结果
    if not all_fold_results:
        return None, None
    
    results_df = pd.DataFrame(all_fold_results)
    avg_metrics = results_df.mean()
    std_metrics = results_df.std()
    
    result = {
        'mode': ablation_mode,
        'acc_mean': avg_metrics.get('accuracy', 0),
        'acc_std': std_metrics.get('accuracy', 0),
        'f1_mean': avg_metrics.get('f1', 0),
        'f1_std': std_metrics.get('f1', 0),
        'auc_mean': avg_metrics.get('auc', 0),
        'auc_std': std_metrics.get('auc', 0),
        'precision_mean': avg_metrics.get('precision', 0),
        'recall_mean': avg_metrics.get('recall', 0),
    }
    
    print(f"\n✅ {ablation_mode}: ACC={result['acc_mean']:.4f}±{result['acc_std']:.4f}, "
          f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}")
    logger.info(f"{ablation_mode}: ACC={result['acc_mean']:.4f}±{result['acc_std']:.4f}, "
                f"F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}")
    
    return result, fold_details


def main():
    parser = argparse.ArgumentParser(description='SEUMLD Ablation Study')
    parser.add_argument('--mode', type=str, default='all',
                        help='Run specific ablation mode or "all"')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (for quick test)')
    args_cli = parser.parse_args()
    
    # 初始化配置
    args = Args()
    
    # 快速测试模式
    if args_cli.epochs:
        args.epochs = args_cli.epochs
    
    # 设置实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_dir = os.path.join(args.exp_dir, f'ablation_{timestamp}')
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # 设置日志
    log_file_path = os.path.join(args.exp_dir, 'ablation_log.txt')
    logger = Logger.setup_logger(log_file_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 打印消融实验设计
    print_ablation_table()
    
    # 确定要运行的模式
    if args_cli.mode == 'all':
        modes_to_run = get_ablation_modes()
    else:
        modes_to_run = [args_cli.mode]
    
    print(f"\n🚀 Running {len(modes_to_run)} ablation configurations...")
    logger.info(f"Running ablation modes: {modes_to_run}")
    
    # 运行所有消融配置
    all_results = []
    all_fold_details = []
    
    for mode in modes_to_run:
        # 使用配置副本，避免修改原始配置
        args_copy = copy.deepcopy(args)
        result, fold_details = run_single_ablation(mode, args_copy, logger, device)
        if result:
            all_results.append(result)
        if fold_details:
            all_fold_details.extend(fold_details)
    
    # 输出对比表格
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 排序 (按 ACC 降序)
        results_df = results_df.sort_values('acc_mean', ascending=False)
        
        # 保存结果
        results_path = os.path.join(args.exp_dir, 'ablation_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # 保存每个fold的详细指标
        if all_fold_details:
            fold_details_df = pd.DataFrame(all_fold_details)
            fold_details_path = os.path.join(args.exp_dir, 'ablation_fold_details.csv')
            fold_details_df.to_csv(fold_details_path, index=False)
        
        print("\n" + "="*100)
        print("📊 ABLATION STUDY RESULTS (sorted by Accuracy)")
        print("="*100)
        print(f"{'Mode':<20} {'ACC':<20} {'F1':<20} {'AUC':<20}")
        print("-"*100)
        
        for _, row in results_df.iterrows():
            acc_str = f"{row['acc_mean']:.4f}±{row['acc_std']:.4f}"
            f1_str = f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}"
            auc_str = f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}"
            print(f"{row['mode']:<20} {acc_str:<20} {f1_str:<20} {auc_str:<20}")
        
        print("="*100)
        
        # 打印每个fold的详细指标
        if all_fold_details:
            print("\n📋 FOLD DETAILS:")
            print("-"*80)
            fold_df = pd.DataFrame(all_fold_details)
            for mode in results_df['mode'].values:
                mode_folds = fold_df[fold_df['mode'] == mode]
                if len(mode_folds) > 0:
                    print(f"\n{mode}:")
                    for _, row in mode_folds.iterrows():
                        print(f"  Fold {row['fold']}: ACC={row['accuracy']:.4f}, F1={row['f1']:.4f}, AUC={row['auc']:.4f}")
        
        print(f"\n📁 Results saved to: {results_path}")
        if all_fold_details:
            print(f"📁 Fold details saved to: {fold_details_path}")
        
        # LaTeX 表格输出
        print("\n📝 LaTeX Table Format:")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Ablation Study Results on SEUMLD Dataset}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("Method & ACC & F1 & AUC \\\\")
        print("\\midrule")
        
        for _, row in results_df.iterrows():
            mode_name = row['mode'].replace('_', ' ').title()
            print(f"{mode_name} & "
                  f"{row['acc_mean']:.2f}$\\pm${row['acc_std']:.2f} & "
                  f"{row['f1_mean']:.2f}$\\pm${row['f1_std']:.2f} & "
                  f"{row['auc_mean']:.2f}$\\pm${row['auc_std']:.2f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        
        logger.info("\n" + results_df.to_string())
    
    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()
