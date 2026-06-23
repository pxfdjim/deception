# import torch.nn as nn
# import torch.nn.functional as F
# import torch


# class VisualInstanceClassifier(nn.Module):
#     """视觉实例分类器 (MIL 结构)"""
#     def __init__(self, input_dim=768, hidden_dim=768, embedding_dim=64, num_classes=2):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(inplace=False),
#             nn.Dropout(0.25)
#         )
#         self.classifier = nn.Linear(hidden_dim, num_classes)
#         self.projector = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=False),
#             nn.Linear(hidden_dim, embedding_dim)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         logits = self.classifier(features)
#         embedding = self.projector(features)
#         embedding = F.normalize(embedding, p=2, dim=1)
#         return logits, embedding


# class AudioGlobalEncoder(nn.Module):
#     """
#     音频全局编码器 (轻量级辅助模态)
#     处理 WavLM 的 1024 维全局声学特征
#     容量刻意小于视觉分支，体现辅助地位
#     """
#     def __init__(self, input_dim=1024, hidden_dim=256, output_dim=128):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, output_dim),
#             nn.LayerNorm(output_dim)
#         )

#     def forward(self, x):
#         return self.network(x)


# class AudioGatedResidual(nn.Module):
#     """
#     视觉主导的门控残差融合
    
#     fused = W_v(v_bag) + gate(v_bag, a_bag) ⊙ W_a(a_bag)
    
#     设计原则:
#       - 视觉是恒等主路径 (即使 gate=0，视觉仍独立工作)
#       - 音频通过学习的门控信号"辅助调节"视觉特征
#       - gate 初始偏置为负值，使训练初期音频贡献接近 0，
#         模型先学好视觉，再逐渐引入音频
#     """
#     def __init__(self, v_dim, a_dim, output_dim):
#         super().__init__()
#         # 视觉主路径投影
#         self.v_proj = nn.Sequential(
#             nn.Linear(v_dim, output_dim),
#             nn.LayerNorm(output_dim),
#             nn.ReLU(inplace=False)
#         )
#         # 音频辅助路径投影
#         self.a_proj = nn.Sequential(
#             nn.Linear(a_dim, output_dim),
#             nn.LayerNorm(output_dim),
#             nn.ReLU(inplace=False)
#         )
#         # 门控网络: 联合决定音频的注入权重
#         self.gate_net = nn.Sequential(
#             nn.Linear(v_dim + a_dim, output_dim),
#             nn.Sigmoid()
#         )
#         # 初始化门控偏置为负值 → 初期 gate ≈ 0，音频贡献极小
#         nn.init.constant_(self.gate_net[0].bias, -2.0)

#     def forward(self, v_bag, a_bag):
#         """
#         Args:
#             v_bag: [B, v_dim] 视觉包级特征 (主)
#             a_bag: [B, a_dim] 音频包级特征 (辅)
#         Returns:
#             fused: [B, output_dim]
#         """
#         v_out = self.v_proj(v_bag)          # 视觉主路径
#         a_out = self.a_proj(a_bag)          # 音频投影
#         gate = self.gate_net(torch.cat([v_bag, a_bag], dim=1))  # [B, output_dim]
#         fused = v_out + gate * a_out        # 残差注入: 视觉 + 门控音频
#         return fused


# class LieDetection(nn.Module):
#     """
#     非对称多模态谎言检测网络 (Visual-Dominant Asymmetric Fusion)
    
#     核心思想: 视觉模态主导，音频作为辅助模态
    
#     支持三种模态模式:
#       - 'visual':  仅视觉 MIL (向后兼容)
#       - 'audio':   仅音频 MLP 全局编码
#       - 'both':    视觉 MIL (主) + 音频 MLP (辅) → 门控残差融合
    
#     视觉 (主): 原型引导 MIL 层次化聚合 (Sequence → Bag)
#     音频 (辅): 轻量级全局声学编码 (Global → Bag)  
#     融合: 门控残差 — fused = v_proj + gate(v,a) ⊙ a_proj
#     """
#     def __init__(self, args, initial_prototypes=None):
#         super().__init__()
#         self.args = args
#         self.modality = getattr(args, 'modality', 'both')

#         # 视觉包级特征维度: K * D
#         self.single_branch_dim = args.low_dim * args.num_classes
#         # 音频编码器输出维度 (可配置，默认等于视觉包维度)
#         self.audio_output_dim = getattr(args, 'audio_output_dim', self.single_branch_dim)

#         if self.modality in ('visual', 'both'):
#             # --- 视觉分支 (MIL) — 主模态 ---
#             self.visual_instance_classifier = VisualInstanceClassifier(
#                 input_dim=getattr(args, 'visual_dim', 768),
#                 hidden_dim=args.hidden_dim,
#                 embedding_dim=args.low_dim, num_classes=args.num_classes
#             )
#             if initial_prototypes is not None:
#                 self.register_buffer("visual_prototypes", initial_prototypes.clone())
#             else:
#                 self.register_buffer("visual_prototypes", torch.zeros(args.num_classes, args.low_dim))

#         if self.modality in ('audio', 'both'):
#             # --- 音频分支 (Global MLP) — 辅助模态, 容量更小 ---
#             audio_hidden = args.hidden_dim // 2  # 音频隐藏层 = 视觉的一半
#             self.audio_encoder = AudioGlobalEncoder(
#                 input_dim=getattr(args, 'audio_dim', 1024),
#                 hidden_dim=audio_hidden,
#                 output_dim=self.audio_output_dim
#             )

#         # --- 分类头 ---
#         if self.modality == 'both':
#             fusion_output_dim = args.hidden_dim
#             self.fusion_module = AudioGatedResidual(
#                 v_dim=self.single_branch_dim,
#                 a_dim=self.audio_output_dim,
#                 output_dim=fusion_output_dim
#             )
#             self.final_classifier = nn.Sequential(
#                 nn.Dropout(0.3),
#                 nn.Linear(fusion_output_dim, args.num_classes)
#             )
#         elif self.modality == 'visual':
#             self.final_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
#         elif self.modality == 'audio':
#             self.final_classifier = nn.Sequential(
#                 nn.Dropout(0.3),
#                 nn.Linear(self.audio_output_dim, args.num_classes)
#             )
#         else:
#             raise ValueError(f"未知的 modality: {self.modality}")

#     # ─────────────────────────────────────────────────
#     #  原型更新（视觉分支专用）
#     # ─────────────────────────────────────────────────
#     @torch.no_grad()
#     def _update_prototypes_single(self, prototypes, embeddings_list,
#                                    bag_labels, instance_predictions_list):
#         """更新视觉原型"""
#         for i in range(len(embeddings_list)):
#             bag_label = bag_labels[i]
#             bag_embeddings = embeddings_list[i]
#             if bag_label == 0:
#                 for emb in bag_embeddings:
#                     prototypes[0] = prototypes[0] * self.args.proto_m + (1 - self.args.proto_m) * emb
#             else:
#                 preds = instance_predictions_list[i]
#                 for j, emb in enumerate(bag_embeddings):
#                     pred_label = preds[j]
#                     prototypes[pred_label] = prototypes[pred_label] * self.args.proto_m + (1 - self.args.proto_m) * emb
#         prototypes.copy_(F.normalize(prototypes, p=2, dim=1))

#     # ─────────────────────────────────────────────────
#     #  视觉 MIL 层次化聚合
#     # ─────────────────────────────────────────────────
#     def _hierarchical_aggregate(self, features_list, instance_classifier, prototypes):
#         """
#         视觉模态: 实例分类 → 原型相似度 → 聚类 → 簇内MaxPool → 簇间拼接
        
#         Args:
#             features_list: list of [N_i, D_in]
#             instance_classifier: VisualInstanceClassifier
#             prototypes: [K, low_dim]
#         Returns:
#             bag_features: [B, K * low_dim]
#             instance_logits_list, instance_embeddings_list
#         """
#         num_instances_per_bag = [f.shape[0] for f in features_list]  # Fix: shape[0]
#         features_flat = torch.cat(features_list, dim=0)

#         instance_logits_flat, instance_embeddings_flat = instance_classifier(features_flat)
#         sim_to_protos_flat = torch.matmul(instance_embeddings_flat, prototypes.t())

#         instance_logits_list = torch.split(instance_logits_flat, num_instances_per_bag, dim=0)
#         instance_embeddings_list = torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0)
#         sim_to_protos_list = torch.split(sim_to_protos_flat, num_instances_per_bag, dim=0)

#         final_bag_features = []  # Fix: 初始化为空列表
#         for i in range(len(instance_embeddings_list)):
#             embeddings = instance_embeddings_list[i]
#             sims = sim_to_protos_list[i]
#             cluster_assignments = torch.argmax(sims, dim=1)

#             cluster_features = []  # Fix: 初始化为空列表
#             for k in range(self.args.num_classes):
#                 indices_in_cluster = (cluster_assignments == k).nonzero(as_tuple=True)[0]  # Fix: 取[0]
#                 if len(indices_in_cluster) == 0:
#                     cluster_feature = torch.zeros(self.args.low_dim, device=embeddings.device)
#                 else:
#                     embeddings_in_cluster = embeddings[indices_in_cluster]
#                     cluster_feature, _ = torch.max(embeddings_in_cluster, dim=0)
#                 cluster_features.append(cluster_feature)

#             final_bag_feature = torch.cat(cluster_features, dim=0)  # [K * low_dim]
#             final_bag_features.append(final_bag_feature)

#         bag_features = torch.stack(final_bag_features, dim=0)  # [B, K * low_dim]
#         return bag_features, list(instance_logits_list), list(instance_embeddings_list)

#     # ─────────────────────────────────────────────────
#     #  Forward
#     # ─────────────────────────────────────────────────
#     def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
#         """
#         Args:
#             visual_features_list: list of [N_i, 768] 或 None
#             audio_features_list:  list of [1, 1024] / Tensor [B, 1024] 或 None
#             bag_labels: [B] 包级标签 (训练时用于更新原型)
#         """
#         if self.modality == 'visual':
#             # ─── 纯视觉 (向后兼容) ───
#             v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(
#                 visual_features_list, self.visual_instance_classifier, self.visual_prototypes)

#             if self.training and bag_labels is not None:
#                 with torch.no_grad():
#                     v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
#                     self._update_prototypes_single(
#                         self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

#             logits = self.final_classifier(v_bag)
#             return {
#                 'logits': logits,
#                 'alignment_loss': torch.tensor(0.0, device=v_bag.device),
#                 'instance_logits_list': v_inst_logits,
#                 'instance_embeddings_list': v_inst_embeds,
#             }

#         elif self.modality == 'audio':
#             # ─── 纯音频 ───
#             if isinstance(audio_features_list, list):
#                 audio_tensor = torch.stack(
#                     [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list], dim=0)
#             else:
#                 audio_tensor = audio_features_list
#             # audio_tensor: (B, audio_dim)
#             a_bag = self.audio_encoder(audio_tensor)
#             logits = self.final_classifier(a_bag)
#             return {
#                 'logits': logits,
#                 'alignment_loss': torch.tensor(0.0, device=a_bag.device),
#                 'instance_logits_list': [],
#                 'instance_embeddings_list': [],
#             }

#         elif self.modality == 'both':
#             # ─── 双模态: 视觉 MIL + 音频 MLP → LRTF 融合 ───
#             # 1. 视觉分支
#             v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(
#                 visual_features_list, self.visual_instance_classifier, self.visual_prototypes)

#             # 更新视觉原型
#             if self.training and bag_labels is not None:
#                 with torch.no_grad():
#                     v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
#                     self._update_prototypes_single(
#                         self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

#             # 2. 音频分支
#             if isinstance(audio_features_list, list):
#                 audio_tensor = torch.stack(
#                     [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list], dim=0)
#             else:
#                 audio_tensor = audio_features_list
#             a_bag = self.audio_encoder(audio_tensor)  # (B, audio_output_dim)

#             # 3. 模态对齐损失 (软正则，不作为强约束)
#             # v_bag 和 a_bag 维度可能不同，投影到共同维度计算
#             align_dim = min(v_bag.shape[1], a_bag.shape[1])
#             alignment_loss = 1.0 - F.cosine_similarity(
#                 v_bag[:, :align_dim], a_bag[:, :align_dim], dim=1).mean()

#             # 4. 门控残差融合: fused = v_proj + gate * a_proj
#             fused_bag = self.fusion_module(v_bag, a_bag)

#             # 5. 分类
#             logits = self.final_classifier(fused_bag)

#             return {
#                 'logits': logits,
#                 'alignment_loss': alignment_loss,
#                 'instance_logits_list': v_inst_logits,
#                 'instance_embeddings_list': v_inst_embeds,
#             }




import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class VisualInstanceClassifier(nn.Module):
    """保留原有视觉实例分类器，用于提取细粒度帧特征"""
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
    """专门为 1024 维全局音频特征设计的编码器"""
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

    def forward(self, visual_instances, audio_global):
        # visual_instances: [N, visual_dim] (某个视频的所有帧特征)
        # audio_global: [1, audio_dim] (该视频的单一音频特征)
        
        Q = self.q_proj(audio_global)      # [1, hidden_dim]
        K = self.k_proj(visual_instances)  # [N, hidden_dim]
        V = self.v_proj(visual_instances)  # [N, hidden_dim]

        # 计算音频对每一帧视觉的注意力分数
        attn_scores = torch.matmul(Q, K.t()) / self.scale  # [1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)      # [1, N]

        # 根据音频的引导，加权聚合视觉帧
        guided_visual_bag = torch.matmul(attn_weights, V)  # [1, hidden_dim]
        return guided_visual_bag, attn_weights


class LieDetection(nn.Module):
    """
    非对称多模态谎言检测网络 (Audio-Guided Visual Attention)

    核心思想: 视觉模态主导，音频作为辅助引导
    - 视觉帧是 Key/Value，音频是 Query
    - 音频决定"关注哪些视觉帧" → 加权聚合 → 拼接音频 → 分类

    支持三种模态模式:
      - 'visual':  仅视觉 MIL (原型引导聚合，向后兼容)
      - 'audio':   仅音频全局编码
      - 'both':    音频引导视觉注意力 + 拼接融合
    """
    def __init__(self, args, initial_prototypes=None):
        super().__init__()
        self.args = args
        self.modality = getattr(args, 'modality', 'both')
        self.single_branch_dim = args.low_dim * args.num_classes  # K * D

        if self.modality in ('visual', 'both'):
            # --- 视觉分支 (MIL) — 主模态 ---
            self.visual_instance_classifier = VisualInstanceClassifier(
                input_dim=getattr(args, 'visual_dim', 768),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim, num_classes=args.num_classes
            )
            if initial_prototypes is not None:
                self.register_buffer("visual_prototypes", initial_prototypes.clone())
            else:
                self.register_buffer("visual_prototypes",
                                     torch.zeros(args.num_classes, args.low_dim))

        if self.modality in ('audio', 'both'):
            # --- 音频分支 (全局 MLP) — 辅助模态 ---
            self.audio_encoder = AudioGlobalEncoder(
                input_dim=getattr(args, 'audio_dim', 1024),
                hidden_dim=args.hidden_dim
            )

        # --- 模态特定的分类头 ---
        if self.modality == 'both':
            # 音频引导注意力: 音频 Query → 视觉帧 K/V → 加权聚合
            self.audio_guided_attn = AudioGuidedAttention(
                visual_dim=args.low_dim,
                audio_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim
            )
            # 将音频引导的视觉特征投影回 v_bag 同维度
            self.guided_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False)
            )
            # 门控网络: 决定音频引导的视觉特征注入多少
            self.audio_gate = nn.Sequential(
                nn.Linear(self.single_branch_dim * 2, self.single_branch_dim),
                nn.Sigmoid()
            )
            # 初始化门控偏置为负值 → 初期 gate ≈ 0，音频几乎不参与
            nn.init.constant_(self.audio_gate[0].bias, -2.0)
            # 最终分类器: 维度 = 纯视觉维度 (K * low_dim)
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim, args.num_classes)
            )
        elif self.modality == 'visual':
            # 纯视觉: 原型 MIL 聚合后直接分类
            self.final_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
        elif self.modality == 'audio':
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(args.hidden_dim, args.num_classes)
            )
        else:
            raise ValueError(f"未知的 modality: {self.modality}")

    # ─────────────────────────────────────────────────
    #  原型更新 (视觉分支专用)
    # ─────────────────────────────────────────────────
    @torch.no_grad()
    def _update_prototypes_single(self, prototypes, embeddings_list,
                                   bag_labels, instance_predictions_list):
        """更新视觉原型"""
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

    # ─────────────────────────────────────────────────
    #  视觉 MIL 层次化聚合 (visual-only 模式用)
    # ─────────────────────────────────────────────────
    def _hierarchical_aggregate(self, features_list):
        """原型引导 MIL: 聚类 → 簇内MaxPool → 拼接"""
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

    # ─────────────────────────────────────────────────
    #  音频输入预处理 (list → tensor)
    # ─────────────────────────────────────────────────
    def _prepare_audio(self, audio_features_list):
        """将训练管线传入的 list/tensor 统一为 [B, audio_dim] 张量"""
        if isinstance(audio_features_list, list):
            return torch.stack(
                [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list], dim=0)
        return audio_features_list

    # ─────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────
    def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
        """
        Args:
            visual_features_list: list of [N_i, 768] 或 None
            audio_features_list:  list of [1024] / Tensor [B, 1024] 或 None
            bag_labels: [B] 包级标签 (训练时用于更新原型)
        """
        if self.modality == 'visual':
            # ─── 纯视觉: 原型引导 MIL 聚合 ───
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(
                visual_features_list)

            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(
                        self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

            logits = self.final_classifier(v_bag)
            return {
                'logits': logits,
                'alignment_loss': torch.tensor(0.0, device=v_bag.device),
                'instance_logits_list': v_inst_logits,
                'instance_embeddings_list': v_inst_embeds,
            }

        elif self.modality == 'audio':
            # ─── 纯音频 ───
            audio_tensor = self._prepare_audio(audio_features_list)
            a_bag = self.audio_encoder(audio_tensor)
            logits = self.final_classifier(a_bag)
            return {
                'logits': logits,
                'alignment_loss': torch.tensor(0.0, device=a_bag.device),
                'instance_logits_list': [],
                'instance_embeddings_list': [],
            }

        elif self.modality == 'both':
            # ─── 双模态: 原型引导 MIL (主) + 音频引导门控残差 (辅) ───
            # 最终特征维度 = K*low_dim = 纯视觉维度，音频不稀释视觉

            # 1. 视觉原型引导 MIL 层次化聚合 (主路径)
            #    聚类 → 簇内MaxPool → 拼接 → v_bag [B, K*low_dim]
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(
                visual_features_list)

            # 更新视觉原型
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(
                        self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

            # 2. 音频全局编码
            audio_tensor = self._prepare_audio(audio_features_list)
            a_global = self.audio_encoder(audio_tensor)  # [B, hidden_dim]

            # 3. 音频引导的视觉注意力 (辅助路径)
            #    音频 Query → 视觉帧 Key/Value → 加权聚合
            B = len(visual_features_list)
            guided_v_list = []
            attn_weights_list = []

            for i in range(B):
                v_inst = v_inst_embeds[i]           # [N_i, low_dim]
                a_feat = a_global[i].unsqueeze(0)   # [1, hidden_dim]

                guided_v, attn_w = self.audio_guided_attn(v_inst, a_feat)
                guided_v_list.append(guided_v)       # [1, hidden_dim]
                attn_weights_list.append(attn_w)

            guided_v_tensor = torch.cat(guided_v_list, dim=0)  # [B, hidden_dim]

            # 4. 门控残差融合: fused = v_bag + gate ⊙ proj(guided_v)
            #    音频不直接出现在最终特征中，只通过门控调节视觉
            guided_v_proj = self.guided_proj(guided_v_tensor)  # [B, K*low_dim]
            gate = self.audio_gate(
                torch.cat([v_bag, guided_v_proj], dim=-1))     # [B, K*low_dim]
            fused_bag = v_bag + gate * guided_v_proj            # [B, K*low_dim]

            # 5. 最终分类
            logits = self.final_classifier(fused_bag)

            return {
                'logits': logits,
                'alignment_loss': torch.tensor(0.0, device=logits.device),
                'instance_logits_list': v_inst_logits,
                'instance_embeddings_list': v_inst_embeds,
                'cross_attention_weights': attn_weights_list,
            }