import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from collections import defaultdict
from sklearn import metrics
from tqdm import tqdm

def compute_metrics(logits, labels):
    """
    计算评估指标：准确率、F1和AUC
    
    Args:
        eval_pred: 可以是(logits, labels)二元组或(true_labels, pred_labels, pred_scores)三元组
    
    Returns:
        包含各种指标的字典
    """
    # print('compute_metrics',logits.shape,labels.shape)
    loss=F.cross_entropy(torch.tensor(logits), torch.tensor(labels).long())
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=-1)
    lie_scores = probs[:, 1]  # 预测为lie类的概率

    # 计算准确率
    acc = accuracy_score(labels, predictions)
    
    # 计算 F1 分数
    f1 = f1_score(labels, predictions, average='binary',zero_division=0)  # 若为二分类    
    
    # 计算 AUC
    if len(np.unique(labels)) == 2 and len(np.unique(predictions)) == 2:
        auc = roc_auc_score(labels, lie_scores)
    else:
        # 如果AUC计算失败(比如只有一个类别)，返回0.5
        print(f"AUC compute failed.")
        auc = 0.5
        
    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "loss": loss.item()
    }



def compute_score(metric, fold_metric_base):
    score=metric["accuracy"]/fold_metric_base["acc"]+metric["f1"]/fold_metric_base["f1"]+metric["auc"]/fold_metric_base["auc"]-metric["loss"]
    return score


def _to_cuda(features_list):
    """将特征列表移到GPU，跳过 None 元素"""
    if features_list is None:
        return None
    return [f.cuda(non_blocking=True) if f is not None else None for f in features_list]


def evaluate_model(model, test_loader, criterion, args):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for visual_list, audio_list, bag_labels, _ in test_loader_tqdm:
            visual_list = _to_cuda(visual_list)
            audio_list = _to_cuda(audio_list)
            bag_labels = bag_labels.cuda(non_blocking=True)
            
            # 前向传播
            outputs = model(visual_list, audio_list, bag_labels=None)
            batch_bag_logits = outputs['logits']
            
            # --- 4. 新增：计算验证损失 ---
            loss = criterion(batch_bag_logits, bag_labels)                                                                       
            total_loss += loss.item()
            num_batches += 1
            
            # ---原有逻辑保持不变---
            batch_bag_probs = F.softmax(batch_bag_logits, dim=-1)
            bag_score_tensor = batch_bag_probs[:, 1] # "说谎"类别的概率
            preds = (bag_score_tensor > 0.50).long()
            
            all_preds.extend(preds.cpu().numpy())
            # 【注意】因为上面 bag_labels 移到了 GPU，这里必须加 .cpu() 才能转 numpy
            all_labels.extend(bag_labels.cpu().numpy()) 
            all_probs.extend(bag_score_tensor.cpu().numpy())

    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 计算指标
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    test_loader_tqdm.close()
    
    try:
        auc = metrics.roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    print(f"\n Evaluation Results:")
    print(f"   Val Loss:  {avg_loss:.4f}") # 打印 Loss
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    return {
        'val_loss': avg_loss, # 返回 Loss
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall,
        'f1': f1, 
        'auc': auc
    }


def evaluate_real_model(model, test_loader, criterion, args):
    """
    评估模型性能（Real-Life数据集版本 - LOOCV）
    
    Args:
        model: 模型
        test_loader: 测试数据加载器（batch_size=1）
        criterion: 损失函数
        args: 参数配置
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    
    with torch.no_grad():
        # LOOCV: 测试集只有一个样本
        for visual_list, audio_list, labels, ids in test_loader:
            visual_list = _to_cuda(visual_list)
            audio_list = _to_cuda(audio_list)
            labels = labels.cuda(non_blocking=True)
            sample_id = ids[0]
            
            # 前向传播
            outputs = model(visual_list, audio_list, labels)
            
            # 计算损失
            loss = criterion(outputs['logits'], labels)
            
            # 获取预测结果
            probs = torch.softmax(outputs['logits'], dim=1)
            pred = torch.argmax(probs, dim=1)
            
            # 转换为标量
            true_label = labels[0].item()
            prediction = pred[0].item()
            probability = probs[0, 1].item()  # 正类概率
            
            # 计算准确率
            accuracy = 1.0 if prediction == true_label else 0.0
            
            # 计算F1（单样本）
            if prediction == 1 and true_label == 1:
                f1 = 1.0
            elif prediction == 0 and true_label == 0:
                f1 = 1.0
            else:
                f1 = 0.0
            
            # 计算AUC（单样本无法计算，返回概率）
            auc = probability if true_label == 1 else (1 - probability)
            
            return {
                'val_loss': loss.item(),
                'loss': loss.item(),  # 兼容性
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'prediction': prediction,
                'true_label': true_label,
                'probability': probability,
                'id': sample_id
            }
    
    # 如果没有数据，返回默认值
    return {
        'val_loss': 0.0,
        'loss': 0.0,
        'accuracy': 0.0,
        'f1': 0.0,
        'auc': 0.0,
        'prediction': 0,
        'true_label': 0,
        'probability': 0.0,
        'id': 'unknown'
    }