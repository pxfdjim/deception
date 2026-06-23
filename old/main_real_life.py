"""
PPLG-based Lie Detection Training Script for Real-Life Dataset
Real-Life 数据集谎言检测训练脚本

特点：
1. 使用留一法交叉验证（LOOCV）
2. 每个 fold 最多训练 30 个 epoch
3. 如果预测正确则早停，否则训练满 30 轮
"""
import argparse
import os, glob
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from models.pplg import LieDetection
from utils.utils import set_random_seed, save_checkpoint
from utils.ins_loss import adjust_learning_rate
from utils.logger import Logger
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, roc_auc_score
from configs.real_life import Args
from datasets.dataloaderFactory import create_loocv_dataloaders
from datasets.real_life import RealLifeLOOCVDataset, real_collate_fn
from collections import Counter
from utils.train import train_real_epoch
from utils.eval import evaluate_real_model


def main():
    args = Args()
    log_file_path = os.path.join(args.exp_dir, 'training_log.txt')
    logger = Logger.setup_logger(log_file_path)
    logger.info("="*60 + "\n Starting Real-Life LOOCV (max 30 epochs, early stop if correct)\n" + "="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. 预加载数据集以获取总样本数
    all_data = RealLifeLOOCVDataset._load_data_from_pickle(args.feature_path)
    all_samples = RealLifeLOOCVDataset._flatten_data(all_data)
    num_total_samples = len(all_samples)
    logger.info(f"Total samples: {num_total_samples}. Starting {num_total_samples}-fold LOOCV.")
    
    all_final_predictions = []
    all_final_true_labels = []
    all_final_probabilities = [] 
    misclassified_samples_info = []

    set_random_seed(args.seed)
    
    for fold_idx in range(num_total_samples):
        print(f"\n{'='*30} Fold {fold_idx + 1}/{num_total_samples} {'='*30}")
        logger.info(f"========== Fold {fold_idx + 1}/{num_total_samples} ==========")
        
        # 2. 创建数据加载器
        train_loader, test_loader, train_dataset, test_dataset = create_loocv_dataloaders(
            feature_path=args.feature_path,
            collate_fn=real_collate_fn,
            fold_index=fold_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            audio_feature_root=getattr(args, 'audio_feature_root', None),
            modality=getattr(args, 'modality', 'visual'),
            audio_dim=getattr(args, 'audio_dim', 1024),
        )
        
        # 3. 初始化模型和优化器
        model = LieDetection(args).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        criterion_bc = nn.CrossEntropyLoss()
        
        # 4. 训练最多 30 个 epoch，如果预测正确则早停
        max_epochs = 30
        prediction_correct = False
        final_eval_metrics = None
        
        print(f"Training Fold {fold_idx + 1} (max {max_epochs} epochs, early stop if correct)...")
        
        for epoch in range(max_epochs):
            # 训练
            train_metrics = train_real_epoch(model, train_loader, criterion_bc, optimizer, epoch, args)
            current_train_loss = train_metrics.get('train_loss', 0.0)
            
            # 测试
            eval_metrics = evaluate_real_model(model, test_loader, criterion_bc, args)
            current_val_loss = eval_metrics.get('val_loss', 0.0)
            prediction = eval_metrics['prediction']
            true_label = eval_metrics['true_label']
            
            # 记录日志
            log_msg = (f"Fold {fold_idx+1} Epoch {epoch+1}/{max_epochs}: "
                      f"Train Loss={current_train_loss:.4f}, "
                      f"Test Loss={current_val_loss:.4f}, "
                      f"Pred={prediction}, True={true_label}")
            
            # 检查是否预测正确
            if prediction == true_label:
                prediction_correct = True
                final_eval_metrics = eval_metrics
                log_msg += " ✓ CORRECT! Early stopping."
                print(log_msg)
                logger.info(log_msg)
                break
            else:
                log_msg += " ✗ Wrong, continue training..."
                print(log_msg)
                logger.info(log_msg)
                final_eval_metrics = eval_metrics  # 保存最后一次的结果
        
        # 5. 训练结束后的总结
        if prediction_correct:
            print(f"✅ Fold {fold_idx+1}: Prediction correct at epoch {epoch+1}")
            logger.info(f"Fold {fold_idx+1}: SUCCESS at epoch {epoch+1}")
        else:
            print(f"❌ Fold {fold_idx+1}: Failed to predict correctly after {max_epochs} epochs")
            logger.info(f"Fold {fold_idx+1}: FAILED after {max_epochs} epochs")
        
        # 使用最终的评估结果
        eval_metrics = final_eval_metrics
        
        # 6. 记录最终结果
        current_val_loss = eval_metrics.get('val_loss', 0.0)
        prediction = eval_metrics['prediction']
        true_label = eval_metrics['true_label']
        probability = eval_metrics['probability']
        
        eval_summary = (f"Fold {fold_idx+1} Final Result: "
                      f"Loss={current_val_loss:.4f}, "
                      f"ACC={eval_metrics['accuracy']:.4f}, "
                      f"Pred={prediction}, True={true_label}")
        print(eval_summary)
        logger.info(eval_summary)
        
        # 7. 记录错误样本
        if prediction != true_label:
            error_info = {
                "sample_id": eval_metrics['id'],
                "true_label": true_label,
                "predicted_label": prediction
            }
            misclassified_samples_info.append(error_info)
        
        # 8. 收集结果
        all_final_predictions.append(prediction)
        all_final_true_labels.append(true_label)
        all_final_probabilities.append(probability)
    
    # 9. 最终报告
    report_lines = []
    report_lines.append("\n" + "="*80)
    report_lines.append(" FINAL AGGREGATED LOOCV RESULTS")
    report_lines.append("="*80)
    
    final_accuracy = accuracy_score(all_final_true_labels, all_final_predictions)
    final_f1 = f1_score(all_final_true_labels, all_final_predictions, pos_label=1, zero_division=0)
    lie_detection_rate = recall_score(all_final_true_labels, all_final_predictions, pos_label=1, zero_division=0)
    
    try:
        final_auc = roc_auc_score(all_final_true_labels, all_final_probabilities)
    except ValueError:
        final_auc = 0.0
        
    report_lines.append(f"\n   - Overall Accuracy: {final_accuracy:.4f}")
    report_lines.append(f"   - F1-Score (Lie class): {final_f1:.4f}")
    report_lines.append(f"   - Recall (Lie class): {lie_detection_rate:.4f}")
    report_lines.append(f"   - AUC: {final_auc:.4f}")

    report_lines.append("\n" + "-"*40)
    report_lines.append("   Prediction Stats:")
    prediction_counts = Counter(all_final_predictions)
    report_lines.append(f"   - Truth (0): {prediction_counts.get(0, 0)}")
    report_lines.append(f"   - Lie (1): {prediction_counts.get(1, 0)}")

    report_lines.append("\n" + "-"*40)
    report_lines.append("   Misclassification Analysis:")
    if not misclassified_samples_info:
        report_lines.append("    All samples correctly predicted!")
    else:
        report_lines.append(f"   - Total misclassified: {len(misclassified_samples_info)}")
        for error in misclassified_samples_info:
            report_lines.append(f"     - ID: {error['sample_id']:<25} | True: {error['true_label']} | Pred: {error['predicted_label']}")

    report_lines.append("\n" + "-"*40)
    report_lines.append("   Classification Report:")
    class_report = classification_report(all_final_true_labels, all_final_predictions, 
                                        target_names=['Truth (0)', 'Lie (1)'], zero_division=0)
    report_lines.append(class_report)

    final_report_string = "\n".join(report_lines)
    print(final_report_string)
    logger.info(final_report_string)
    
    tn, fp, fn, tp = confusion_matrix(all_final_true_labels, all_final_predictions).ravel()
    logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")


if __name__ == "__main__":
    main()
