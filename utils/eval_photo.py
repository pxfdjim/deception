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
        eval_pred: 可以�?logits, labels)二元组或(true_labels, pred_labels, pred_scores)三元�?
    
    Returns:
        包含各种指标的字�?
    """
    # print('compute_metrics',logits.shape,labels.shape)
    loss=F.cross_entropy(torch.tensor(logits), torch.tensor(labels).long())
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=-1)
    lie_scores = probs[:, 1]  # 预测为lie类的概率

    # 计算准确�?
    acc = accuracy_score(labels, predictions)
    
    # 计算 F1 分数
    f1 = f1_score(labels, predictions, average='binary',zero_division=0)  # 若为二分�?   
    
    # 计算 AUC
    if len(np.unique(labels)) == 2 and len(np.unique(predictions)) == 2:
        auc = roc_auc_score(labels, lie_scores)
    else:
        # 如果AUC计算失败(比如只有一个类�?，返�?.5
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
    """将特征列表移到GPU，跳�?None 元素"""
    if features_list is None:
        return None
    return [f.cuda(non_blocking=True) if f is not None else None for f in features_list]


def evaluate_model(model, test_loader, criterion, args, save_plot=False, target_vid=None):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    
    # 收集用于画图的数�?
    all_cos_sims = []
    all_cos_sims_no_loss = []
    # 保存各类中最典型的案例（根据模型置信度挑选）
    best_tp = {'prob': -1.0, 'attn': None, 'vis': None}  # 最自信�?True Positive
    best_fp = {'prob': -1.0, 'attn': None, 'vis': None}  # 最自信�?False Positive (信誓旦旦误判)
    best_fn = {'prob': 2.0,  'attn': None, 'vis': None}  # 最自信�?False Negative (信誓旦旦漏判)
    
    # 调试标志
    debug_printed = False
    tp_count = 0
    
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for visual_list, audio_list, bag_labels, ids in test_loader_tqdm:
            visual_list = _to_cuda(visual_list)
            audio_list = _to_cuda(audio_list)
            bag_labels = bag_labels.cuda(non_blocking=True)
            
            # 前向传播
            outputs = model(visual_list, audio_list, bag_labels=None)
            batch_bag_logits = outputs['logits']
            
            # 收集所有的相似�?(如果存在)，用于画�?
            if 'cos_sim' in outputs:
                all_cos_sims.append(outputs['cos_sim'].cpu().numpy())
            if 'cos_sim_no_loss' in outputs:
                all_cos_sims_no_loss.append(outputs['cos_sim_no_loss'].cpu().numpy())
            
            # --- 4. 新增：计算验证损�?---
            loss = criterion(batch_bag_logits, bag_labels)                                                                       
            total_loss += loss.item()
            num_batches += 1
            
            # ---原有逻辑保持不变---
            batch_bag_probs = F.softmax(batch_bag_logits, dim=-1)
            bag_score_tensor = batch_bag_probs[:, 1] # "说谎"类别的概�?
            preds = (bag_score_tensor > 0.50).long()
            
            # 动态挑选最典型的案例供画图分析
            if 'inst_logits' in outputs and 'cross_attention_weights' in outputs:
                for b_idx in range(len(bag_labels)):
                    l = bag_labels[b_idx].item()
                    p = preds[b_idx].item()
                    prob = bag_score_tensor[b_idx].item()
                    
                    attn_w = outputs['cross_attention_weights'][b_idx].squeeze().cpu().numpy()
                    vis_logits = outputs['inst_logits'][b_idx]
                    vis_prob = torch.softmax(vis_logits, dim=-1)[:, 1].cpu().numpy()
                    
                    if target_vid is not None:
                        # 强制指定样本模式
                        if ids[b_idx] == target_vid:
                            best_tp.update({'prob': prob, 'attn': attn_w, 'vis': vis_prob, 'vid': ids[b_idx]})
                            print(f"[Debug] Found target video {target_vid}: label={l}, pred={p}, prob={prob:.4f}")
                    else:
                        # 自动挑选最置信模式
                        if l == 1 and p == 1 and prob > best_tp['prob']:    # TP
                            best_tp.update({'prob': prob, 'attn': attn_w, 'vis': vis_prob, 'vid': ids[b_idx]})
                            tp_count += 1
                            
                    if l == 0 and p == 1 and prob > best_fp['prob']:  # FP
                        best_fp.update({'prob': prob, 'attn': attn_w, 'vis': vis_prob, 'vid': ids[b_idx]})
                    elif l == 1 and p == 0 and prob < best_fn['prob']:  # FN
                        best_fn.update({'prob': prob, 'attn': attn_w, 'vis': vis_prob, 'vid': ids[b_idx]})
            else:
                if save_plot and not debug_printed:  # 只打印一次
                    print(f"[Debug] Model outputs missing required keys. Available keys: {outputs.keys()}")
                    debug_printed = True
            all_preds.extend(preds.cpu().numpy())
            # 【注意】因为上�?bag_labels 移到�?GPU，这里必须加 .cpu() 才能�?numpy
            all_labels.extend(bag_labels.cpu().numpy()) 
            all_probs.extend(bag_score_tensor.cpu().numpy())

    # --- 收集画图需要的数据放到返回字典中，不再落盘 ---
    plot_data = {}
    if save_plot:
        print(f"[Debug] Collected samples - TP: prob={best_tp['prob']:.4f}, vid={best_tp.get('vid', 'None')}, attn_shape={best_tp['attn'].shape if best_tp['attn'] is not None else 'None'}")
        print(f"[Debug] Total TP samples found: {tp_count}")
        plot_data = {
            'sim_with_loss': np.concatenate(all_cos_sims, axis=0) if len(all_cos_sims) > 0 else None,
            'sim_no_loss': np.concatenate(all_cos_sims_no_loss, axis=0) if len(all_cos_sims_no_loss) > 0 else None,
            'tp_attn': best_tp['attn'],
            'vis_response': best_tp['vis'],
            'tp_prob': best_tp['prob'],
            'tp_vid': best_tp.get('vid', 'Unknown'),
            'fp_attn': best_fp['attn'],
            'fp_prob': best_fp['prob'],
            'fn_attn': best_fn['attn'],
            'fn_prob': best_fn['prob']
        }
    # -----------------------------------------------------

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
        'auc': auc,
        'plot_data': plot_data
    }


def evaluate_real_model(model, test_loader, criterion, args):
    """
    评估模型性能（Real-Life数据集版�?- LOOCV�?
    
    Args:
        model: 模型
        test_loader: 测试数据加载器（batch_size=1�?
        criterion: 损失函数
        args: 参数配置
    
    Returns:
        dict: 包含各种评估指标的字�?
    """
    model.eval()
    
    with torch.no_grad():
        # LOOCV: 测试集只有一个样�?
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
            
            # 转换为标�?
            true_label = labels[0].item()
            prediction = pred[0].item()
            probability = probs[0, 1].item()  # 正类概率
            
            # 计算准确�?
            accuracy = 1.0 if prediction == true_label else 0.0
            
            # 计算F1（单样本�?
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
                'loss': loss.item(),  # 兼容�?
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'prediction': prediction,
                'true_label': true_label,
                'probability': probability,
                'id': sample_id
            }
    
    # 如果没有数据，返回默认�?
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
