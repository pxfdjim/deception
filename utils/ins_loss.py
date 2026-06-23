"""
INS损失函数 - 从INS项目适配到谎言检测任务

主要包含：
1. partial_loss: 处理偏标签的损失函数
2. SupConLoss: 监督对比学习损失
3. confidence更新机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def soft_cross_entropy_loss(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1)
    return loss.mean()
class PartialLoss(nn.Module):
    """
    部分标签损失函数
    
    基于INS中的partial_loss实现，用于处理噪声和部分标签
    包含置信度更新机制
    """
    
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.clone()
        
        # EMA相关参数
        self.conf_ema_m = 0.999
        self.conf_ema_range = [0.95, 0.8]  # (start, end)
    
    def set_conf_ema_m(self, epoch, args):
        """根据epoch设置置信度EMA系数"""
        start = self.conf_ema_range[0]  # 0.95
        end = self.conf_ema_range[1]    # 0.8
        
        # 线性衰减
        self.conf_ema_m = start * (end / start) ** (epoch / args.epochs)
        
    def confidence_update(self, temp_un_conf, batch_index, batchY, point):
        """
        更新置信度
        
        Args:
            temp_un_conf: 原型网络的输出 [batch_size, num_classes]
            batch_index: 批次索引
            batchY: 批次的部分标签 [batch_size, num_classes]
            point: 掩码点
        """
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)  # 获取原型预测
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float()
            
            # EMA更新置信度
            self.confidence[batch_index, :] = \
                self.conf_ema_m * self.confidence[batch_index, :] + \
                (1 - self.conf_ema_m) * pseudo_label
                
            # 确保在有效标签空间内
            self.confidence[batch_index, :] *= batchY
            
            # 标准化
            row_sum = self.confidence[batch_index, :].sum(dim=1, keepdim=True)
            self.confidence[batch_index, :] /= row_sum
    
    def forward(self, outputs, index):
        """
        前向传播
        
        Args:
            outputs: 模型输出 [batch_size, num_classes]
            index: 样本索引
        """
        # 取对数softmax
        logsm_outputs = F.log_softmax(outputs, dim=1)
        
        # 使用当前置信度计算损失
        final_outputs = logsm_outputs * self.confidence[index, :]
        
        # 负对数似然损失
        average_loss = -torch.sum(final_outputs) / final_outputs.size(0)
        
        return average_loss


class SupConLoss(nn.Module):
    """
    监督对比学习损失 - 从INS适配
    
    基于MoCo的队列机制和INS的伪标签对比策略
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, mask=None, batch_size=None):
        """
        计算监督对比损失
        
        Args:
            features: 特征张量 [2*batch_size + queue_size, feature_dim]
                     包含查询特征、关键字特征和队列特征
            mask: 正样本掩码 [batch_size, batch_size + queue_size]
                  如果为None，则使用MoCo样式的对比学习（warmup阶段）
            batch_size: 批次大小
        """
        device = features.device
        
        if mask is not None:
            # INS阶段：使用伪标签构建正负样本对
            return self._ins_contrastive_loss(features, mask, batch_size)
        else:
            # Warmup阶段：使用MoCo样式的对比学习
            return self._moco_contrastive_loss(features, batch_size)
    
    def _ins_contrastive_loss(self, features, mask, batch_size):
        """INS阶段的对比学习损失"""
        # 分离查询特征
        device = features.device
        q_features = features[:batch_size]  # [batch_size, feature_dim]
        
        # 计算相似度
        # features: [2*batch_size + queue_size, feature_dim]
        anchor_dot_contrast = torch.div(
            torch.matmul(q_features, features.T),
            self.temperature
        )  # [batch_size, 2*batch_size + queue_size]
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 对角掩码：去除自身
        # diag_mask = torch.eye(batch_size, device=device)
        # mask = mask * (1 - diag_mask)
        
        # 计算正样本的对数概率
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 正样本平均对数概率
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # 损失
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _moco_contrastive_loss(self, features, batch_size):
        """MoCo样式的对比学习损失（warmup阶段）"""
        # 查询和关键字特征
        q = features[:batch_size]           # [batch_size, feature_dim]
        k = features[batch_size:2*batch_size]  # [batch_size, feature_dim]
        queue = features[2*batch_size:]     # [queue_size, feature_dim]
        
        # 正样本：查询和对应的关键字
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [batch_size, 1]
        
        # 负样本：查询和队列
        l_neg = torch.einsum('nc,ck->nk', [q, queue.T])  # [batch_size, queue_size]
        
        # 组合logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, 1 + queue_size]
        logits /= self.temperature
        
        # 标签：正样本在第0位
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    可选的损失函数，用于替代标准交叉熵
    """
    
    def __init__(self, confidence, alpha=1, gamma=2):
        super().__init__()
        self.confidence = confidence
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, outputs, index):
        """计算focal loss"""
        # Softmax概率
        probs = F.softmax(outputs, dim=1)
        
        # 取对数
        log_probs = F.log_softmax(outputs, dim=1)
        
        # 使用置信度加权
        weighted_log_probs = log_probs * self.confidence[index, :]
        
        # Focal weight: (1 - p)^gamma
        focal_weights = torch.pow(1 - probs, self.gamma)
        
        # 最终损失
        focal_loss = -self.alpha * focal_weights * weighted_log_probs
        
        return focal_loss.sum() / outputs.size(0)


class ClsLoss(nn.Module):
    """
    分类损失 - 用于袋级监督
    """
    
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
    
    def forward(self, outputs, labels):
        """
        Args:
            outputs: [batch_size, num_classes]
            labels: [batch_size] - 真实标签
        """
        return F.cross_entropy(outputs, labels)


# 工具函数
def accuracy(output, target, topk=(1,)):
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(args, optimizer, epoch):
    """学习率调度"""
    import math
    
    lr = args.lr
    
    if args.cosine:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.lr_decay_epochs:
            lr *= args.lr_decay_rate if epoch >= milestone else 1.
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
