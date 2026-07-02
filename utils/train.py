
import torch
from utils.ins_loss import AverageMeter, adjust_learning_rate
import time
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def _to_cuda(features_list):
    """将特征列表移到GPU，跳过 None 元素"""
    if features_list is None:
        return None
    return [f.cuda(non_blocking=True) if f is not None else None for f in features_list]


# def train_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
#     model.train()
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f') 
#     losses = AverageMeter('Loss', ':.4e')
#     bag_main_losses = AverageMeter('MainBagLoss', ':.4e')
    
#     end = time.time()
#     train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
#     for batch_idx, (features_list, bag_labels, indices) in enumerate(train_loader_tqdm):
#         data_time.update(time.time() - end)
        
#         # 将数据移到GPU
#         features_list = [f.cuda(non_blocking=True) for f in features_list]
#         bag_labels = bag_labels.cuda(non_blocking=True)
#         B = len(features_list)
        
     
#         outputs = model(features_list, bag_labels)
        
#         # --- 计算包级别损失 (这部分保持不变) ---
#         loss_bag_main = criterion_bc(outputs['bag_logits'], bag_labels)

#         total_loss = args.lambda_bag_attn * loss_bag_main 
#         # --- 反向传播和优化 ---
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
        
#         # --- 记录统计信息 ---
#         losses.update(total_loss.item(), B)
#         bag_main_losses.update(loss_bag_main.item(), B)
        
#         batch_time.update(time.time() - end)
#         end = time.time()
        
#         train_loader_tqdm.set_postfix(
#             Loss=losses.avg, 
#             MainBag=bag_main_losses.avg,
#         )
#     train_loader_tqdm.close()
    
#     return {
#         'train_loss': losses.avg,
#         'main_bag_loss': bag_main_losses.avg,
# #     }

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    alpha: 类别权重
    gamma: 聚焦参数，越大越关注难分样本
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def weak_instance_loss(instance_logits_list, bag_labels, topk_ratio=0.25):
    losses = []
    for instance_logits, bag_label in zip(instance_logits_list, bag_labels):
        if instance_logits.numel() == 0:
            continue

        if int(bag_label.item()) == 0:
            pseudo_labels = torch.zeros(
                instance_logits.size(0),
                dtype=torch.long,
                device=instance_logits.device
            )
            losses.append(F.cross_entropy(instance_logits, pseudo_labels))
        else:
            num_pos = max(1, int(round(instance_logits.size(0) * topk_ratio)))
            lie_scores = F.softmax(instance_logits.detach(), dim=-1)[:, 1]
            top_indices = torch.topk(lie_scores, k=min(num_pos, instance_logits.size(0))).indices
            pseudo_labels = torch.ones(
                top_indices.size(0),
                dtype=torch.long,
                device=instance_logits.device
            )
            losses.append(F.cross_entropy(instance_logits[top_indices], pseudo_labels))

    if not losses:
        device = bag_labels.device if torch.is_tensor(bag_labels) else "cuda"
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def mil_evidence_loss(
    instance_logits_list,
    bag_labels,
    topk_ratio=0.25,
    rank_margin=0.5,
    rank_weight=0.05,
):
    bag_logits = []
    targets = []
    rank_losses = []

    for instance_logits, bag_label in zip(instance_logits_list, bag_labels):
        if instance_logits.numel() == 0:
            continue

        lie_scores = instance_logits[:, 1] - instance_logits[:, 0]
        num_top = max(1, int(round(lie_scores.numel() * topk_ratio)))
        num_top = min(num_top, lie_scores.numel())
        top_scores = torch.topk(lie_scores, k=num_top, largest=True).values
        bag_logits.append(top_scores.mean())
        targets.append(bag_label.to(dtype=lie_scores.dtype))

        if int(bag_label.item()) == 1 and lie_scores.numel() >= 2:
            num_bottom = min(num_top, lie_scores.numel() - num_top)
            if num_bottom > 0:
                bottom_scores = torch.topk(lie_scores, k=num_bottom, largest=False).values
                rank_losses.append(F.relu(rank_margin - (top_scores.mean() - bottom_scores.mean())))

    if not bag_logits:
        device = bag_labels.device if torch.is_tensor(bag_labels) else "cuda"
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    bag_logits = torch.stack(bag_logits)
    targets = torch.stack(targets).to(device=bag_logits.device)
    evidence_loss = F.binary_cross_entropy_with_logits(bag_logits, targets)
    if rank_losses and rank_weight > 0:
        rank_loss = torch.stack(rank_losses).mean()
    else:
        rank_loss = torch.tensor(0.0, device=bag_logits.device)
    return evidence_loss + float(rank_weight) * rank_loss, rank_loss


def train_new(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    训练一个epoch（优化版：移除KL损失，添加实例级监督）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion_bc: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        args: 参数配置
    
    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    bag_losses = AverageMeter('BagLoss', ':.4e')
    ins_losses = AverageMeter('InsLoss', ':.4e')
    
    train_loader_tqdm = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{args.epochs} Training",
        disable=getattr(args, "disable_tqdm", False)
    )
    
    # Warmup学习率
    if hasattr(args, 'warmup_epochs') and args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        warmup_lr_scale = (epoch + 1) / args.warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * warmup_lr_scale
    
    # 实例级损失权重：逐渐增加
    ins_weight = min(0.3, 0.1 * (epoch + 1) / 10)  # 前10个epoch从0.1增加到0.3
    
    for batch_idx, (visual_list, audio_list, bag_labels, video_names) in enumerate(train_loader_tqdm):
        # 将数据移到GPU
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)
        
        # 前向传播
        outputs = model(visual_list, audio_list, bag_labels)
        
        # 1. 包级损失（主要损失）
        bag_loss = criterion_bc(outputs['logits'], bag_labels)
        ins_loss = 0.0
        if hasattr(args, 'use_instance_loss') and args.use_instance_loss:
            instance_logits_list = outputs['instance_logits_list']
            for i, instance_logits in enumerate(instance_logits_list):
                bag_label = bag_labels[i]
                # 为实例生成伪标签
                if bag_label == 0:
                    # 真实样本：所有实例都是真实
                    pseudo_labels = torch.zeros(instance_logits.size(0), dtype=torch.long, device=instance_logits.device)
                else:
                    # 谎言样本：使用模型预测作为伪标签
                    with torch.no_grad():
                        pseudo_labels = torch.argmax(instance_logits, dim=1)
                
                ins_loss += F.cross_entropy(instance_logits, pseudo_labels)
            
            ins_loss = ins_loss / len(instance_logits_list)
        
        # 总损失
        total_loss = args.lambda_bag_attn * bag_loss + ins_weight * ins_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 记录统计信息
        losses.update(total_loss.item(), B)
        bag_losses.update(bag_loss.item(), B)
        if isinstance(ins_loss, torch.Tensor):
            ins_losses.update(ins_loss.item(), B)
        
        train_loader_tqdm.set_postfix(
            Loss=losses.avg, 
            BagLoss=bag_losses.avg,
            InsLoss=ins_losses.avg
        )
    
    train_loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
        'bag_loss': bag_losses.avg,
        'ins_loss': ins_losses.avg,
    }



def train_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    训练一个epoch（通用版本）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion_bc: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        args: 参数配置
    
    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
    for batch_idx, (visual_list, audio_list, bag_labels, video_names) in enumerate(train_loader_tqdm):
        # 将数据移到GPU
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)
        
        # 前向传播
        outputs = model(visual_list, audio_list, bag_labels)
        
        # 计算损失
        total_loss = criterion_bc(outputs['logits'], bag_labels)
        # +outputs['kl_loss']
        # total_loss = args.lambda_main * loss 
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录统计信息
        losses.update(total_loss.item(), B)
        
        train_loader_tqdm.set_postfix(Loss=losses.avg)
    
    train_loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,  # 兼容性
    }
# def train_epoch(model, train_loader, compute_loss, optimizer, epoch, args):
#     model.train()
#     losses = AverageMeter('Loss', ':.4e')
    
#     train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
#     for batch_idx, (features_list, bag_labels, video_names) in enumerate(train_loader_tqdm):
#         # 1. 数据准备
#         features_list = [f.cuda(non_blocking=True) for f in features_list]
#         bag_labels = bag_labels.cuda(non_blocking=True)
#         B = len(features_list)
        
#         # 2. 前向传播 (Forward)
#         outputs = model(features_list, bag_labels)
        
#         # 3. 计算损失 (Loss)
#         # 注意：这里调用之前定义的 compute_loss
#         total_loss = compute_loss(outputs, bag_labels)
        
#         # 4. 反向传播 (Backward)
#         optimizer.zero_grad()
#         total_loss.backward()
        
#         # 5. 梯度裁剪 (可选但推荐)
#         if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
#         # 6. 参数更新 (Optimizer Step)
#         optimizer.step()
        
#         # 7. !!! 关键修改：原型更新移到这里 !!!
#         # 此时梯度计算已完成，修改 prototypes 不会影响 backward
#         model.update_prototypes(outputs, bag_labels)
        
#         # 8. 记录
#         losses.update(total_loss.item(), B)
#         train_loader_tqdm.set_postfix(Loss=losses.avg)
    
#     train_loader_tqdm.close()
    
#     return {
#         'train_loss': losses.avg,
#         'total_loss': losses.avg, 
#     }

def train_real_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    训练一个epoch（Real-Life数据集版本）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion_bc: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        args: 参数配置
    
    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
    for batch_idx, (visual_list, audio_list, bag_labels, ids) in enumerate(train_loader_tqdm):
        # 将数据移到GPU
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)
        
        # 前向传播
        outputs = model(visual_list, audio_list, bag_labels)
        
        # 计算损失
        loss = criterion_bc(outputs['logits'], bag_labels)
        train_loss = args.lambda_main * loss
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录统计信息
        losses.update(train_loss.item(), B)
        
        train_loader_tqdm.set_postfix(Loss=losses.avg)
    
    
    train_loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,  # 兼容性
    }


# ============================================================
#  NewData 专用: 被试内 Pairwise Ranking + 分类联合训练
# ============================================================
def train_newdata_ranking_epoch(model, subject_loader, criterion_bc, optimizer, epoch, args):
    """
    NewData (NumberGuess) 被试内排序训练
    
    使用 SubjectDataset 的 subject_loader，每个被试有20道题:
    - 分类损失 (CE): 对20道题各自做二分类
    - 排序损失 (MarginRanking): 谎话题分数 > 真话题分数 + margin
    
    两个损失联合优化，解决训练(分类)和评估(Top-K排序)的目标不一致问题。
    
    Args:
        model: LieDetection 模型
        subject_loader: SubjectDataset 的 DataLoader (返回 5-tuple)
        criterion_bc: 分类损失函数 (CrossEntropyLoss)
        optimizer: 优化器
        epoch: 当前 epoch
        args: 参数配置 (需包含 rank_margin, rank_weight)
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    rank_losses = AverageMeter('RankLoss', ':.4e')
    
    rank_margin = getattr(args, 'rank_margin', 1.0)
    rank_weight = getattr(args, 'rank_weight', 0.5)
    
    loader_tqdm = tqdm(subject_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
    for batch_idx, (v_feats, a_feats, labels, qids, metas) in enumerate(loader_tqdm):
        # v_feats: (B, max_q, max_seg_v, v_dim)
        # a_feats: (B, max_q, max_seg_a, a_dim)  
        # labels: (B, max_q), -1 = padding
        B = v_feats.shape[0]
        
        total_ce = torch.tensor(0.0, device='cuda')
        total_rank = torch.tensor(0.0, device='cuda')
        n_valid = 0
        
        for i in range(B):
            # 提取该被试的有效问题 (排除padding)
            valid_mask = labels[i] != -1
            valid_v = v_feats[i][valid_mask]       # (n_q, max_seg_v, v_dim)
            valid_a = a_feats[i][valid_mask]       # (n_q, max_seg_a, a_dim)
            valid_labels = labels[i][valid_mask]   # (n_q,)
            
            n_q = valid_v.shape[0]
            if n_q == 0:
                continue
            
            # 转为 features_list 格式，送入模型
            visual_list = [valid_v[j].cuda() for j in range(n_q)]
            audio_list = [valid_a[j].cuda() for j in range(n_q)]
            bag_labels = valid_labels.cuda()
            
            # 前向传播: 20道题作为20个"包"
            outputs = model(visual_list, audio_list, bag_labels)
            logits = outputs['logits']  # (n_q, 2)
            
            # ① 分类损失: 每道题的 CE
            ce_loss = criterion_bc(logits, bag_labels)
            total_ce = total_ce + ce_loss
            
            # ② 被试内排序损失: lie的分数应 > truth的分数 + margin
            lie_mask = bag_labels == 1
            truth_mask = bag_labels == 0
            lie_scores = logits[lie_mask, 1]       # 谎话题的 lie 类概率
            truth_scores = logits[truth_mask, 1]   # 真话题的 lie 类概率
            
            if lie_scores.numel() > 0 and truth_scores.numel() > 0:
                # 构造所有 (lie, truth) 配对
                n_lie = lie_scores.shape[0]
                n_truth = truth_scores.shape[0]
                lie_expanded = lie_scores.unsqueeze(1).expand(n_lie, n_truth).reshape(-1)
                truth_expanded = truth_scores.unsqueeze(0).expand(n_lie, n_truth).reshape(-1)
                target = torch.ones_like(lie_expanded)  # lie 应该 > truth
                
                rank_loss = F.margin_ranking_loss(lie_expanded, truth_expanded, target, margin=rank_margin)
                total_rank = total_rank + rank_loss
            
            n_valid += 1
        
        if n_valid == 0:
            continue
        
        # 平均损失
        avg_ce = total_ce / n_valid
        avg_rank = total_rank / n_valid
        total_loss = avg_ce + rank_weight * avg_rank
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # 记录
        losses.update(total_loss.item(), n_valid)
        ce_losses.update(avg_ce.item(), n_valid)
        if isinstance(avg_rank, torch.Tensor) and avg_rank.item() > 0:
            rank_losses.update(avg_rank.item(), n_valid)
        
        loader_tqdm.set_postfix(
            Loss=f"{losses.avg:.4f}",
            CE=f"{ce_losses.avg:.4f}",
            Rank=f"{rank_losses.avg:.4f}"
        )
    
    loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
        'ce_loss': ce_losses.avg,
        'rank_loss': rank_losses.avg,
    }





# ============================================================
#  pplg_muti 模型: 通用训练函数 (DOLOS / 通用数据集)
# ============================================================
def train_muti_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    pplg_muti 模型训练: CE
    DataLoader 返回 4-tuple: (visual_list, audio_list, bag_labels, names)
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")

    for batch_idx, (visual_list, audio_list, bag_labels, _) in enumerate(train_loader_tqdm):
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)

        outputs = model(visual_list, audio_list, bag_labels)

        total_loss = criterion_bc(outputs['logits'], bag_labels)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.update(total_loss.item(), B)

        train_loader_tqdm.set_postfix(Loss=f"{losses.avg:.4f}")

    train_loader_tqdm.close()
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
    }
def train_new_epo(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    pplg_muti 模型训练: CE + 音频辅助 Loss + 正交正则化 Loss
    DataLoader 返回 4-tuple: (visual_list, audio_list, bag_labels, names)
    """
    model.train()
    model.current_epoch = epoch
    
    # 用多个 Meter 分别记录各项 Loss，方便排查问题
    losses = AverageMeter('Loss', ':.4e')
    losses_cls = AverageMeter('Loss_Cls', ':.4e')
    # losses_aux = AverageMeter('Loss_Aux', ':.4e')
    losses_ort = AverageMeter('Loss_Ort', ':.4e')
    losses_inst = AverageMeter('Loss_Inst', ':.4e')
    losses_loop = AverageMeter('Loss_Loop', ':.4e')
    losses_margin = AverageMeter('Loss_Margin', ':.4e')
    losses_rank = AverageMeter('Loss_Rank', ':.4e')
    losses_mil = AverageMeter('Loss_MIL', ':.4e')
    losses_mil_rank = AverageMeter('Loss_MILRank', ':.4e')
    instance_loss_weight = getattr(args, 'instance_loss_weight', 0.1)
    positive_instance_topk_ratio = getattr(args, 'positive_instance_topk_ratio', 0.25)
    proto_loop_loss_weight = getattr(args, 'proto_loop_loss_weight', 0.05)

    train_loader_tqdm = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{args.epochs} Training",
        disable=getattr(args, "disable_tqdm", False)
    )

    for batch_idx, (visual_list, audio_list, bag_labels, _) in enumerate(train_loader_tqdm):
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)

        outputs = model(visual_list, audio_list, bag_labels)

        # ==========================================
        # 1. 计算主分类 Loss
        # ==========================================
        ce_loss = criterion_bc(outputs['logits'], bag_labels)
        total_loss = ce_loss
        
        # 2. 计算正交正则化 Loss (如果存在)
        ortho_loss = torch.tensor(0.0, device=bag_labels.device)
        if 'ortho_loss' in outputs:
            ortho_loss = outputs['ortho_loss']
            ortho_weight = getattr(args, 'ortho_weight', 0.1)
            total_loss = total_loss + ortho_weight * ortho_loss

        instance_loss = torch.tensor(0.0, device=bag_labels.device)
        if getattr(args, 'use_instance_loss', True) and 'instance_logits_list' in outputs:
            instance_loss = weak_instance_loss(
                outputs['instance_logits_list'],
                bag_labels,
                topk_ratio=positive_instance_topk_ratio
            )
            total_loss = total_loss + instance_loss_weight * instance_loss

        proto_loop_loss = torch.tensor(0.0, device=bag_labels.device)
        if getattr(args, 'use_proto_loop_consistency', False) and 'proto_loop_loss' in outputs:
            proto_loop_loss = outputs['proto_loop_loss']
            total_loss = total_loss + proto_loop_loss_weight * proto_loop_loss

        logit_margin_loss = torch.tensor(0.0, device=bag_labels.device)
        margin_warmup_epochs = getattr(args, 'logit_margin_warmup_epochs', 10)
        if (
            getattr(args, 'use_logit_margin_regularization', False)
            and epoch >= margin_warmup_epochs
        ):
            logits = outputs['logits']
            target_margin = float(getattr(args, 'logit_margin_target', 3.0))
            logit_margins = torch.abs(logits[:, 1] - logits[:, 0])
            logit_margin_loss = F.relu(logit_margins - target_margin).pow(2).mean()
            total_loss = total_loss + float(getattr(args, 'logit_margin_weight', 0.02)) * logit_margin_loss

        batch_rank_loss = torch.tensor(0.0, device=bag_labels.device)
        rank_warmup_epochs = getattr(args, 'batch_rank_warmup_epochs', 5)
        if getattr(args, 'use_batch_rank_loss', False) and epoch >= rank_warmup_epochs:
            lie_scores = outputs['logits'][:, 1] - outputs['logits'][:, 0]
            pos_scores = lie_scores[bag_labels == 1]
            neg_scores = lie_scores[bag_labels == 0]
            if pos_scores.numel() > 0 and neg_scores.numel() > 0:
                rank_margin = float(getattr(args, 'batch_rank_margin', 0.5))
                pairwise_gap = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                batch_rank_loss = F.relu(rank_margin - pairwise_gap).mean()
                total_loss = total_loss + float(getattr(args, 'batch_rank_loss_weight', 0.05)) * batch_rank_loss

        mil_loss = torch.tensor(0.0, device=bag_labels.device)
        mil_rank_loss = torch.tensor(0.0, device=bag_labels.device)
        mil_warmup = getattr(args, 'mil_evidence_warmup_epochs', 0)
        if (
            getattr(args, 'use_mil_evidence_loss', False)
            and epoch >= mil_warmup
            and 'instance_logits_list' in outputs
        ):
            mil_loss, mil_rank_loss = mil_evidence_loss(
                outputs['instance_logits_list'],
                bag_labels,
                topk_ratio=getattr(args, 'mil_evidence_topk_ratio', 0.25),
                rank_margin=getattr(args, 'mil_evidence_rank_margin', 0.5),
                rank_weight=getattr(args, 'mil_evidence_rank_weight', 0.05),
            )
            total_loss = total_loss + float(getattr(args, 'mil_evidence_loss_weight', 0.1)) * mil_loss

        # # ==========================================
        # # 3. 计算特征正交正则化 Loss (Orthogonal Loss)
        # # 惩罚音频学到和视觉一样的东西，逼它找新线索
        # # ==========================================
        # loss_ortho = torch.tensor(0.0, device=bag_labels.device)
        # if 'guided_v_proj' in outputs and 'v_bag_detached' in outputs:
        #     v_feat = outputs['v_bag_detached']
        #     a_feat = outputs['guided_v_proj']

        #     # L2 归一化
        #     v_feat_norm = F.normalize(v_feat, p=2, dim=1)
        #     a_feat_norm = F.normalize(a_feat, p=2, dim=1)

        #     # 计算 Batch 内的余弦相似度绝对值的均值
        #     # 越接近 0 越好(正交)，所以把这个相似度当惩罚项加进 Loss
        #     cos_sim = torch.abs(torch.sum(v_feat_norm * a_feat_norm, dim=1))
        #     loss_ortho = cos_sim.mean()

        #     # 权重建议 0.1
        #     total_loss += args.ortho_weight * loss_ortho

        # ==========================================
        # 反向传播 & 优化
        # ==========================================
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 更新统计
        losses.update(total_loss.item(), B)
        losses_cls.update(ce_loss.item(), B)
        if isinstance(ortho_loss, torch.Tensor) and ortho_loss.item() > 0:
            losses_ort.update(ortho_loss.item(), B)
        if isinstance(instance_loss, torch.Tensor) and instance_loss.item() > 0:
            losses_inst.update(instance_loss.item(), B)
        if isinstance(proto_loop_loss, torch.Tensor) and proto_loop_loss.item() > 0:
            losses_loop.update(proto_loop_loss.item(), B)
        if isinstance(logit_margin_loss, torch.Tensor) and logit_margin_loss.item() > 0:
            losses_margin.update(logit_margin_loss.item(), B)
        if isinstance(batch_rank_loss, torch.Tensor) and batch_rank_loss.item() > 0:
            losses_rank.update(batch_rank_loss.item(), B)
        if isinstance(mil_loss, torch.Tensor) and mil_loss.item() > 0:
            losses_mil.update(mil_loss.item(), B)
        if isinstance(mil_rank_loss, torch.Tensor) and mil_rank_loss.item() > 0:
            losses_mil_rank.update(mil_rank_loss.item(), B)

        # 在进度条里拆分显示，让你对网络内部状态一目了然
        if not getattr(args, "disable_tqdm", False):
            train_loader_tqdm.set_postfix({
                'Tot': f"{losses.avg:.3f}",
                'CE': f"{losses_cls.avg:.3f}",
                'Inst': f"{losses_inst.avg:.3f}",
                'Loop': f"{losses_loop.avg:.3f}",
                'Ort': f"{losses_ort.avg:.3f}",
                'Margin': f"{losses_margin.avg:.3f}",
                'Rank': f"{losses_rank.avg:.3f}",
                'MIL': f"{losses_mil.avg:.3f}",
                'MILR': f"{losses_mil_rank.avg:.3f}",
            })

    train_loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'ce_loss': losses_cls.avg,
        'instance_loss': losses_inst.avg,
        'proto_loop_loss': losses_loop.avg,
        'ortho_loss': losses_ort.avg,
        'logit_margin_loss': losses_margin.avg,
        'batch_rank_loss': losses_rank.avg,
        'mil_evidence_loss': losses_mil.avg,
        'mil_evidence_rank_loss': losses_mil_rank.avg,
    }


def train_muti_real_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    pplg_muti 模型 Real-Life 版: CE
    DataLoader 返回 4-tuple: (visual_list, audio_list, bag_labels, ids)
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")

    for batch_idx, (visual_list, audio_list, bag_labels, ids) in enumerate(train_loader_tqdm):
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)

        outputs = model(visual_list, audio_list, bag_labels)

        total_loss = criterion_bc(outputs['logits'], bag_labels)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), B)
        train_loader_tqdm.set_postfix(Loss=f"{losses.avg:.4f}")

    train_loader_tqdm.close()
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
    }


# ============================================================
#  pplg_muti 模型: NewData 被试内 Ranking + CE
# ============================================================
def train_muti_newdata_ranking_epoch(model, subject_loader, criterion_bc, optimizer, epoch, args):
    """
    pplg_muti 模型 NewData 版: CE + Ranking
    subject_loader 返回 5-tuple: (v_feats, a_feats, labels, qids, metas)
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    ce_losses_m = AverageMeter('CE', ':.4e')
    rank_losses_m = AverageMeter('Rank', ':.4e')

    rank_margin = getattr(args, 'rank_margin', 1.0)
    rank_weight = getattr(args, 'rank_weight', 0.5)

    loader_tqdm = tqdm(subject_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")

    for batch_idx, (v_feats, a_feats, labels, qids, metas) in enumerate(loader_tqdm):
        B = v_feats.shape[0]

        total_ce = torch.tensor(0.0, device='cuda')
        total_rank = torch.tensor(0.0, device='cuda')
        n_valid = 0

        for i in range(B):
            valid_mask = labels[i] != -1
            valid_v = v_feats[i][valid_mask]
            valid_a = a_feats[i][valid_mask]
            valid_labels = labels[i][valid_mask]

            n_q = valid_v.shape[0]
            if n_q == 0:
                continue

            visual_list = [valid_v[j].cuda() for j in range(n_q)]
            audio_list = [valid_a[j].cuda() for j in range(n_q)]
            bag_labels = valid_labels.cuda()

            outputs = model(visual_list, audio_list, bag_labels)
            logits = outputs['logits']

            # ① CE
            ce_loss = criterion_bc(logits, bag_labels)
            total_ce = total_ce + ce_loss

            # ② Pairwise Ranking
            lie_mask = bag_labels == 1
            truth_mask = bag_labels == 0
            lie_scores = logits[lie_mask, 1]
            truth_scores = logits[truth_mask, 1]

            if lie_scores.numel() > 0 and truth_scores.numel() > 0:
                n_lie = lie_scores.shape[0]
                n_truth = truth_scores.shape[0]
                lie_expanded = lie_scores.unsqueeze(1).expand(n_lie, n_truth).reshape(-1)
                truth_expanded = truth_scores.unsqueeze(0).expand(n_lie, n_truth).reshape(-1)
                target = torch.ones_like(lie_expanded)

                rank_loss = F.margin_ranking_loss(lie_expanded, truth_expanded, target, margin=rank_margin)
                total_rank = total_rank + rank_loss

            n_valid += 1

        if n_valid == 0:
            continue

        avg_ce = total_ce / n_valid
        avg_rank = total_rank / n_valid
        total_loss = avg_ce + rank_weight * avg_rank

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.update(total_loss.item(), n_valid)
        ce_losses_m.update(avg_ce.item(), n_valid)
        if isinstance(avg_rank, torch.Tensor) and avg_rank.item() > 0:
            rank_losses_m.update(avg_rank.item(), n_valid)

        loader_tqdm.set_postfix(
            Loss=f"{losses.avg:.4f}",
            CE=f"{ce_losses_m.avg:.4f}",
            Rank=f"{rank_losses_m.avg:.4f}"
        )

    loader_tqdm.close()

    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
        'ce_loss': ce_losses_m.avg,
        'rank_loss': rank_losses_m.avg,
    }

def train_muti_newdata_epoch(model, subject_loader, criterion_bc, optimizer, epoch, args):
    """
    pplg_muti 模型 NewData 版: CE + Ranking + 音频辅助 Loss + 特征正交 Loss
    subject_loader 返回 5-tuple: (v_feats, a_feats, labels, qids, metas)
    """
    model.train()
    
    # 增加追踪各项 Loss 的 Meter
    losses = AverageMeter('Loss', ':.4e')
    ce_losses_m = AverageMeter('CE', ':.4e')
    rank_losses_m = AverageMeter('Rank', ':.4e')
    ort_losses_m = AverageMeter('Ort', ':.4e')

    # 获取权重超参数
    rank_margin = getattr(args, 'rank_margin', 1.0)
    rank_weight = getattr(args, 'rank_weight', 0.5)
    ortho_weight = getattr(args, 'ortho_weight', 0.1)  # 特征正交惩罚权重

    loader_tqdm = tqdm(subject_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")

    for batch_idx, (v_feats, a_feats, labels, qids, metas) in enumerate(loader_tqdm):
        B = v_feats.shape[0]

        total_ce = torch.tensor(0.0, device='cuda')
        total_rank = torch.tensor(0.0, device='cuda')
        total_ortho = torch.tensor(0.0, device='cuda')
        n_valid = 0

        for i in range(B):
            valid_mask = labels[i] != -1
            valid_v = v_feats[i][valid_mask]
            valid_a = a_feats[i][valid_mask]
            valid_labels = labels[i][valid_mask]

            n_q = valid_v.shape[0]
            if n_q == 0:
                continue

            visual_list = [valid_v[j].cuda() for j in range(n_q)]
            audio_list = [valid_a[j].cuda() for j in range(n_q)]
            bag_labels = valid_labels.cuda()

            # 前向传播
            outputs = model(visual_list, audio_list, bag_labels)
            logits = outputs['logits']

            # ① 主分类 CE Loss
            ce_loss = criterion_bc(logits, bag_labels)
            total_ce = total_ce + ce_loss

            # ② Pairwise Ranking Loss
            lie_mask = bag_labels == 1
            truth_mask = bag_labels == 0
            lie_scores = logits[lie_mask, 1]
            truth_scores = logits[truth_mask, 1]

            if lie_scores.numel() > 0 and truth_scores.numel() > 0:
                n_lie = lie_scores.shape[0]
                n_truth = truth_scores.shape[0]
                lie_expanded = lie_scores.unsqueeze(1).expand(n_lie, n_truth).reshape(-1)
                truth_expanded = truth_scores.unsqueeze(0).expand(n_lie, n_truth).reshape(-1)
                target = torch.ones_like(lie_expanded)

                rank_loss = F.margin_ranking_loss(lie_expanded, truth_expanded, target, margin=rank_margin)
                total_rank = total_rank + rank_loss

            # ③ 特征正交正则化 Orthogonal Loss
            if 'ortho_loss' in outputs:
                total_ortho = total_ortho + outputs['ortho_loss']

            n_valid += 1

        if n_valid == 0:
            continue

        # 对有效样本求平均
        avg_ce = total_ce / n_valid
        avg_rank = total_rank / n_valid if isinstance(total_rank, torch.Tensor) else 0.0
        avg_ortho = total_ortho / n_valid if isinstance(total_ortho, torch.Tensor) else 0.0

        # 组合总 Loss 进行回传
        total_loss = avg_ce + rank_weight * avg_rank + ortho_weight * avg_ortho

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 更新仪表盘数据
        losses.update(total_loss.item(), n_valid)
        ce_losses_m.update(avg_ce.item(), n_valid)
        
        if isinstance(avg_rank, torch.Tensor) and avg_rank.item() > 0:
            rank_losses_m.update(avg_rank.item(), n_valid)
            
        if isinstance(avg_ortho, torch.Tensor) and avg_ortho.item() > 0:
            ort_losses_m.update(avg_ortho.item(), n_valid)

        # 打印到终端
        loader_tqdm.set_postfix(
            Tot=f"{losses.avg:.3f}",
            CE=f"{ce_losses_m.avg:.3f}",
            Rnk=f"{rank_losses_m.avg:.3f}",
            Ort=f"{ort_losses_m.avg:.3f}"
        )

    loader_tqdm.close()
    
    # 返回训练指标
    return {
        'train_loss': losses.avg,
        'ce_loss': ce_losses_m.avg,
        'rank_loss': rank_losses_m.avg,
        'ortho_loss': ort_losses_m.avg
    }


# ============================================================
#  SEUMLD 数据集专用训练函数
# ============================================================
def train_seumld_epoch(model, train_loader, criterion_bc, optimizer, epoch, args):
    """
    SEUMLD 数据集训练: CE + 正交正则化 Loss
    DataLoader 返回 4-tuple: (visual_list, audio_list, bag_labels, video_names)
    
    Args:
        model: LieDetection 模型
        train_loader: 训练数据加载器
        criterion_bc: 分类损失函数
        optimizer: 优化器
        epoch: 当前 epoch
        args: 参数配置
    
    Returns:
        dict: 包含训练指标的字典
    """
    model.train()
    model.current_epoch = epoch
    
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    ortho_losses = AverageMeter('OrthoLoss', ':.4e')
    instance_losses = AverageMeter('InstanceLoss', ':.4e')
    proto_loop_losses = AverageMeter('ProtoLoopLoss', ':.4e')
    
    ortho_weight = getattr(args, 'ortho_weight', 0.1)
    instance_loss_weight = getattr(args, 'instance_loss_weight', 0.1)
    positive_instance_topk_ratio = getattr(args, 'positive_instance_topk_ratio', 0.25)
    proto_loop_loss_weight = getattr(args, 'proto_loop_loss_weight', 0.05)
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
    
    for batch_idx, (visual_list, audio_list, bag_labels, _) in enumerate(train_loader_tqdm):
        visual_list = _to_cuda(visual_list)
        audio_list = _to_cuda(audio_list)
        bag_labels = bag_labels.cuda(non_blocking=True)
        B = bag_labels.size(0)
        
        # 前向传播
        outputs = model(visual_list, audio_list, bag_labels)
        
        # 1. 主分类损失
        ce_loss = criterion_bc(outputs['logits'], bag_labels)
        
        # 2. 正交正则化损失（如果模型支持）
        ortho_loss = torch.tensor(0.0, device='cuda')
        if 'ortho_loss' in outputs:
            ortho_loss = outputs['ortho_loss']

        instance_loss = torch.tensor(0.0, device=bag_labels.device)
        if getattr(args, 'use_instance_loss', True) and 'instance_logits_list' in outputs:
            instance_loss = weak_instance_loss(
                outputs['instance_logits_list'],
                bag_labels,
                topk_ratio=positive_instance_topk_ratio
            )

        proto_loop_loss = torch.tensor(0.0, device=bag_labels.device)
        if getattr(args, 'use_proto_loop_consistency', False) and 'proto_loop_loss' in outputs:
            proto_loop_loss = outputs['proto_loop_loss']
        
        # 总损失
        total_loss = (
            ce_loss
            + instance_loss_weight * instance_loss
            + proto_loop_loss_weight * proto_loop_loss
        )
        # + ortho_weight * ortho_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # 记录统计信息
        losses.update(total_loss.item(), B)
        ce_losses.update(ce_loss.item(), B)
        if isinstance(ortho_loss, torch.Tensor) and ortho_loss.item() > 0:
            ortho_losses.update(ortho_loss.item(), B)
        if isinstance(instance_loss, torch.Tensor) and instance_loss.item() > 0:
            instance_losses.update(instance_loss.item(), B)
        if isinstance(proto_loop_loss, torch.Tensor) and proto_loop_loss.item() > 0:
            proto_loop_losses.update(proto_loop_loss.item(), B)
        
        if not getattr(args, "disable_tqdm", False):
            train_loader_tqdm.set_postfix(
                Loss=f"{losses.avg:.4f}",
                CE=f"{ce_losses.avg:.4f}",
                Inst=f"{instance_losses.avg:.4f}",
                Loop=f"{proto_loop_losses.avg:.4f}",
                Ortho=f"{ortho_losses.avg:.4f}"
            )
    
    train_loader_tqdm.close()
    
    return {
        'train_loss': losses.avg,
        'total_loss': losses.avg,
        'ce_loss': ce_losses.avg,
        'instance_loss': instance_losses.avg,
        'proto_loop_loss': proto_loop_losses.avg,
        'ortho_loss': ortho_losses.avg,
    }
