"""
多模态融合消融实验模型
======================

消融模式说明:
- 'full': 完整模型 (baseline)
- 'only_visual': 仅视觉模态
- 'only_audio': 仅音频模态
- 'no_attention': 无跨模态注意力 (直接投影音频特征)
- 'no_gate': 无门控机制 (直接加权求和)
- 'no_rezero': 无 ReZero 参数 (固定融合系数)
- 'no_audio_content': 无音频内容直连 (仅保留注意力引导)
- 'no_audio_drop': 无 Audio Drop 正则化
- 'early_fusion': 早期融合 (特征级拼接后分类)
- 'late_fusion': 后期融合 (独立分类器 + 概率融合)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class VisualInstanceClassifier(nn.Module):
    """视觉实例分类器，用于提取细粒度帧特征"""
    def __init__(self, input_dim=768, hidden_dim=768, embedding_dim=64, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        embedding = self.projector(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return logits, embedding


class AudioGlobalEncoder(nn.Module):
    """音频全局编码器"""
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        return self.network(x)


class AudioGuidedAttention(nn.Module):
    """跨模态注意力：用全局音频特征引导视觉序列"""
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_proj = nn.Linear(visual_dim, hidden_dim)
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, visual_instances, audio_global):
        Q = self.q_proj(audio_global)
        K = self.k_proj(visual_instances)
        V = self.v_proj(visual_instances)
        temp = torch.clamp(self.attn_temp, min=0.01)
        attn_scores = (torch.matmul(Q, K.t()) / self.scale) / temp
        attn_weights = F.softmax(attn_scores, dim=-1)
        guided_visual_bag = torch.matmul(attn_weights, V)
        return guided_visual_bag, attn_weights


class LieDetectionAblation(nn.Module):
    """
    多模态融合消融实验模型
    
    通过 fusion_mode 参数控制不同的融合策略
    """
    
    def __init__(self, args, initial_prototypes=None):
        super().__init__()
        self.args = args
        self.fusion_mode = getattr(args, 'fusion_mode', 'full')
        self.single_branch_dim = args.low_dim * args.num_classes  # K * D
        
        print(f"[Ablation Model] Fusion Mode: {self.fusion_mode}")
        
        # ============ 视觉分支 ============
        self.visual_instance_classifier = VisualInstanceClassifier(
            input_dim=getattr(args, 'visual_dim', 768),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.low_dim,
            num_classes=args.num_classes
        )
        
        if initial_prototypes is not None:
            self.register_buffer("visual_prototypes", initial_prototypes.clone())
        else:
            self.register_buffer("visual_prototypes",
                                 torch.zeros(args.num_classes, args.low_dim))
        
        # ============ 音频分支 ============
        self.audio_encoder = AudioGlobalEncoder(
            input_dim=getattr(args, 'audio_dim', 1024),
            hidden_dim=args.hidden_dim
        )
        
        # ============ 融合模块 (根据消融模式配置) ============
        
        # 跨模态注意力 (no_attention 模式下不使用)
        if self.fusion_mode != 'no_attention':
            self.audio_guided_attn = AudioGuidedAttention(
                visual_dim=args.low_dim,
                audio_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim
            )
            self.guided_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False)
            )
        
        # 音频内容投影 (no_audio_content / only_visual / only_audio 模式下不使用)
        # 注意: no_attention 模式需要 audio_proj，因为没有跨模态注意力引导
        if self.fusion_mode not in ('no_audio_content', 'only_visual', 'only_audio'):
            self.audio_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False)
            )
        
        # 门控网络 (no_gate / only_visual / only_audio / early_fusion / late_fusion 模式下不使用)
        if self.fusion_mode not in ('no_gate', 'only_visual', 'only_audio', 'early_fusion', 'late_fusion'):
            # 根据是否有音频内容直连和跨模态注意力，调整输入维度
            if self.fusion_mode == 'no_audio_content':
                # 无音频内容直连，但有跨模态注意力: v_bag + guided_v_proj
                gate_input_dim = self.single_branch_dim * 2
            elif self.fusion_mode == 'no_attention':
                # 无跨模态注意力，但有音频投影: v_bag + a_proj
                gate_input_dim = self.single_branch_dim * 2
            else:
                # 完整模式: v_bag + guided_v_proj + a_proj
                gate_input_dim = self.single_branch_dim * 3
            
            self.audio_gate = nn.Sequential(
                nn.Linear(gate_input_dim, self.single_branch_dim),
                nn.Sigmoid()
            )
            nn.init.constant_(self.audio_gate[0].bias, -5.0)
        
        # ReZero 参数 (no_rezero / early_fusion / late_fusion 模式下不使用)
        if self.fusion_mode not in ('no_rezero', 'only_visual', 'only_audio', 'early_fusion', 'late_fusion'):
            self.audio_alpha = nn.Parameter(torch.zeros(self.single_branch_dim))
        
        # 融合 Dropout
        self.fusion_dropout = nn.Dropout(0.5)
        
        # ============ 分类器 ============
        if self.fusion_mode == 'early_fusion':
            # 早期融合: 拼接后分类
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim * 2, args.num_classes)
            )
        elif self.fusion_mode == 'late_fusion':
            # 后期融合: 独立分类器
            self.visual_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
            self.audio_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim, args.num_classes)
            )
        elif self.fusion_mode == 'only_audio':
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(args.hidden_dim, args.num_classes)
            )
        else:
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim, args.num_classes)
            )
    
    @torch.no_grad()
    def _update_prototypes_single(self, prototypes, embeddings_list,
                                  bag_labels, instance_predictions_list):
        for i in range(len(embeddings_list)):
            bag_label = bag_labels[i]
            bag_embeddings = embeddings_list[i]
            if bag_label == 0:
                for emb in bag_embeddings:
                    prototypes[0] = prototypes[0] * self.args.proto_m + (1 - self.args.proto_m) * emb
            else:
                preds = instance_predictions_list[i]
                for j, emb in enumerate(bag_embeddings):
                    pred_label = preds[j]
                    prototypes[pred_label] = (prototypes[pred_label] * self.args.proto_m
                                              + (1 - self.args.proto_m) * emb)
        prototypes.copy_(F.normalize(prototypes, p=2, dim=1))
    
    def _hierarchical_aggregate(self, features_list):
        prototypes = self.visual_prototypes
        num_instances_per_bag = [f.shape[0] for f in features_list]
        features_flat = torch.cat(features_list, dim=0)

        instance_logits_flat, instance_embeddings_flat = self.visual_instance_classifier(features_flat)
        sim_to_protos_flat = torch.matmul(instance_embeddings_flat, prototypes.t())

        instance_logits_list = torch.split(instance_logits_flat, num_instances_per_bag, dim=0)
        instance_embeddings_list = torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0)
        sim_to_protos_list = torch.split(sim_to_protos_flat, num_instances_per_bag, dim=0)

        final_bag_features = []
        for i in range(len(instance_embeddings_list)):
            embeddings = instance_embeddings_list[i]
            sims = sim_to_protos_list[i]
            cluster_assignments = torch.argmax(sims, dim=1)

            cluster_features = []
            for k in range(self.args.num_classes):
                indices_in_cluster = (cluster_assignments == k).nonzero(as_tuple=True)[0]
                if len(indices_in_cluster) == 0:
                    cluster_feature = torch.zeros(self.args.low_dim, device=embeddings.device)
                else:
                    cluster_feature, _ = torch.max(embeddings[indices_in_cluster], dim=0)
                cluster_features.append(cluster_feature)
            final_bag_features.append(torch.cat(cluster_features, dim=0))

        bag_features = torch.stack(final_bag_features, dim=0)
        return bag_features, list(instance_logits_list), list(instance_embeddings_list)
    
    def _prepare_audio(self, audio_features_list):
        if isinstance(audio_features_list, list):
            return torch.stack(
                [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list], dim=0)
        return audio_features_list
    
    def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
        """
        根据 fusion_mode 执行不同的前向传播
        """
        
        # ============ 模式: only_visual ============
        if self.fusion_mode == 'only_visual':
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
            logits = self.final_classifier(v_bag)
            return {'logits': logits}
        
        # ============ 模式: only_audio ============
        if self.fusion_mode == 'only_audio':
            audio_tensor = self._prepare_audio(audio_features_list)
            a_bag = self.audio_encoder(audio_tensor)
            logits = self.final_classifier(a_bag)
            return {'logits': logits}
        
        # ============ 多模态融合模式 ============
        
        # 1. 视觉特征提取
        v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)
        if self.training and bag_labels is not None:
            with torch.no_grad():
                v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
        
        # 2. 音频特征提取
        audio_tensor = self._prepare_audio(audio_features_list)
        a_global = self.audio_encoder(audio_tensor)
        
        # ============ 模式: early_fusion (早期融合) ============
        if self.fusion_mode == 'early_fusion':
            # 直接拼接视觉和音频特征
            a_proj = self.audio_proj(a_global)
            fused = torch.cat([v_bag, a_proj], dim=-1)
            logits = self.final_classifier(fused)
            return {'logits': logits}
        
        # ============ 模式: late_fusion (后期融合) ============
        if self.fusion_mode == 'late_fusion':
            # 独立分类后概率融合
            a_proj = self.audio_proj(a_global)
            v_logits = self.visual_classifier(v_bag)
            a_logits = self.audio_classifier(a_proj)
            
            # 概率融合 (平均)
            v_prob = F.softmax(v_logits, dim=-1)
            a_prob = F.softmax(a_logits, dim=-1)
            fused_prob = (v_prob + a_prob) / 2
            logits = torch.log(fused_prob + 1e-8)  # 转回 logit
            
            return {'logits': logits, 'v_logits': v_logits, 'a_logits': a_logits}
        
        # ============ 模式: no_attention (无跨模态注意力) ============
        if self.fusion_mode == 'no_attention':
            # 直接投影音频，不使用注意力引导
            a_proj = self.audio_proj(a_global)
            
            # 门控融合: v_bag (512) + a_proj (512) -> gate (512)
            gate_input = torch.cat([v_bag, a_proj], dim=-1)
            gate = self.audio_gate(gate_input)
            audio_residual = self.audio_alpha * gate * self.fusion_dropout(a_proj)
            
            # Audio Drop
            if self.training and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)
            
            fused_bag = v_bag + audio_residual
            logits = self.final_classifier(fused_bag)
            
            return {'logits': logits}
        
        # ============ 其他融合模式 (需要跨模态注意力) ============
        
        # 3. 跨模态注意力引导
        B = len(visual_features_list)
        guided_v_list = []
        attn_weights_list = []
        
        for i in range(B):
            v_inst = v_inst_embeds[i]
            a_feat = a_global[i].unsqueeze(0)
            guided_v, attn_w = self.audio_guided_attn(v_inst, a_feat)
            guided_v_list.append(guided_v)
            attn_weights_list.append(attn_w)
        
        guided_v_tensor = torch.cat(guided_v_list, dim=0)
        guided_v_proj = self.guided_proj(guided_v_tensor)
        
        # 4. 音频内容投影 (可选)
        if self.fusion_mode != 'no_audio_content':
            a_proj = self.audio_proj(a_global)
        else:
            a_proj = None
        
        # ============ 模式: no_audio_content (无音频内容直连) ============
        if self.fusion_mode == 'no_audio_content':
            gate_input = torch.cat([v_bag.detach(), guided_v_proj], dim=-1)
            gate = self.audio_gate(gate_input)
            audio_residual = self.audio_alpha * gate * self.fusion_dropout(guided_v_proj)
            
            if self.training and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)
            
            fused_bag = v_bag + audio_residual
            logits = self.final_classifier(fused_bag)
            
            return {'logits': logits, 'cross_attention_weights': attn_weights_list}
        
        # ============ 模式: no_gate (无门控机制) ============
        if self.fusion_mode == 'no_gate':
            # 直接使用固定的融合系数
            combined_audio_info = guided_v_proj + a_proj
            # 使用 ReZero 但不用门控，固定权重 0.5
            audio_residual = self.audio_alpha * 0.5 * self.fusion_dropout(combined_audio_info)
            
            if self.training and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)
            
            fused_bag = v_bag + audio_residual
            logits = self.final_classifier(fused_bag)
            
            return {'logits': logits, 'cross_attention_weights': attn_weights_list}
        
        # ============ 模式: no_rezero (无 ReZero 参数) ============
        if self.fusion_mode == 'no_rezero':
            gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
            gate = self.audio_gate(gate_input)
            combined_audio_info = guided_v_proj + a_proj
            # 固定 alpha = 1.0
            audio_residual = 1.0 * gate * self.fusion_dropout(combined_audio_info)
            
            if self.training and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)
            
            fused_bag = v_bag + audio_residual
            logits = self.final_classifier(fused_bag)
            
            return {'logits': logits, 'cross_attention_weights': attn_weights_list}
        
        # ============ 模式: no_audio_drop (无 Audio Drop) ============
        if self.fusion_mode == 'no_audio_drop':
            gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
            gate = self.audio_gate(gate_input)
            combined_audio_info = guided_v_proj + a_proj
            audio_residual = self.audio_alpha * gate * self.fusion_dropout(combined_audio_info)
            # 不进行 Audio Drop
            
            fused_bag = v_bag + audio_residual
            logits = self.final_classifier(fused_bag)
            
            return {'logits': logits, 'cross_attention_weights': attn_weights_list}
        
        # ============ 模式: full (完整模型) ============
        gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
        gate = self.audio_gate(gate_input)
        combined_audio_info = guided_v_proj + a_proj
        audio_residual = self.audio_alpha * gate * self.fusion_dropout(combined_audio_info)
        
        # Audio Drop
        if self.training and torch.rand(1).item() < 0.2:
            audio_residual = torch.zeros_like(audio_residual)
        
        fused_bag = v_bag + audio_residual
        logits = self.final_classifier(fused_bag)
        
        # 正交损失
        v_feat_norm = F.normalize(v_bag.detach(), p=2, dim=1)
        a_feat_norm = F.normalize(guided_v_proj, p=2, dim=1)
        cos_sim = torch.abs(torch.sum(v_feat_norm * a_feat_norm, dim=1))
        ortho_loss = cos_sim.mean()
        
        return {
            'logits': logits,
            'cross_attention_weights': attn_weights_list,
            'ortho_loss': ortho_loss
        }


def get_ablation_modes():
    """返回所有支持的消融模式"""
    return [
        'full',              # 完整模型
        'only_visual',       # 仅视觉
        'only_audio',        # 仅音频
        'no_attention',      # 无跨模态注意力
        'no_gate',           # 无门控机制
        'no_rezero',         # 无 ReZero 参数
        'no_audio_content',  # 无音频内容直连
        'no_audio_drop',     # 无 Audio Drop
        'early_fusion',      # 早期融合
        'late_fusion',       # 后期融合
    ]


def print_ablation_table():
    """打印消融实验设计表格"""
    print("\n" + "="*80)
    print("多模态融合消融实验设计")
    print("="*80)
    print(f"{'模式':<20} {'说明':<60}")
    print("-"*80)
    modes = {
        'full': '完整模型 (Baseline)',
        'only_visual': '仅视觉模态',
        'only_audio': '仅音频模态',
        'no_attention': '无跨模态注意力 (直接投影音频特征)',
        'no_gate': '无门控机制 (固定权重 0.5)',
        'no_rezero': '无 ReZero 参数 (固定 alpha=1.0)',
        'no_audio_content': '无音频内容直连 (仅保留注意力引导)',
        'no_audio_drop': '无 Audio Drop 正则化',
        'early_fusion': '早期融合 (特征拼接后分类)',
        'late_fusion': '后期融合 (独立分类器 + 概率融合)',
    }
    for mode, desc in modes.items():
        print(f"{mode:<20} {desc:<60}")
    print("="*80 + "\n")


if __name__ == '__main__':
    print_ablation_table()
