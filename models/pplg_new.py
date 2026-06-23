# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import math

# class VisualInstanceClassifier(nn.Module):
#     """保留原有视觉实例分类器，用于提取细粒度帧特征"""
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
#     """专门为 1024 维全局音频特征设计的编码器"""
#     def __init__(self, input_dim=1024, hidden_dim=256):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.3)
#         )
#     def forward(self, x):
#         return self.network(x)

# class AudioGuidedAttention(nn.Module):
#     """跨模态注意力：用全局音频特征引导视觉序列 (带有温度锐化)"""
#     def __init__(self, visual_dim, audio_dim, hidden_dim):
#         super().__init__()
#         self.q_proj = nn.Linear(audio_dim, hidden_dim)
#         self.k_proj = nn.Linear(visual_dim, hidden_dim)
#         self.v_proj = nn.Linear(visual_dim, hidden_dim)
#         self.scale = math.sqrt(hidden_dim)
        
#         # [优化点] 注意力温度锐化，迫使注意力更集中
#         self.attn_temp = nn.Parameter(torch.tensor(0.5))

#     def forward(self, visual_instances, audio_global):
#         Q = self.q_proj(audio_global)      # [1, hidden_dim]
#         K = self.k_proj(visual_instances)  # [N, hidden_dim]
#         V = self.v_proj(visual_instances)  # [N, hidden_dim]

#         temp = torch.clamp(self.attn_temp, min=0.01)
        
#         attn_scores = (torch.matmul(Q, K.t()) / self.scale) / temp  # [1, N]
#         attn_weights = F.softmax(attn_scores, dim=-1)               # [1, N]

#         guided_visual_bag = torch.matmul(attn_weights, V)  # [1, hidden_dim]
#         return guided_visual_bag, attn_weights


# class LieDetection(nn.Module):
#     """
#     非对称多模态谎言检测网络 - 终极安全融合版
#     包含: 内容直连、辅助分类器、通道级 ReZero、温度锐化、梯度截断、Audio Drop
#     """
#     def __init__(self, args, initial_prototypes=None):
#         super().__init__()
#         self.args = args
#         self.modality = getattr(args, 'modality', 'both')
#         self.single_branch_dim = args.low_dim * args.num_classes  # K * D

#         if self.modality in ('visual', 'both'):
#             self.visual_instance_classifier = VisualInstanceClassifier(
#                 input_dim=getattr(args, 'visual_dim', 768),
#                 hidden_dim=args.hidden_dim,
#                 embedding_dim=args.low_dim, num_classes=args.num_classes
#             )
#             if initial_prototypes is not None:
#                 self.register_buffer("visual_prototypes", initial_prototypes.clone())
#             else:
#                 self.register_buffer("visual_prototypes",
#                                      torch.zeros(args.num_classes, args.low_dim))

#         if self.modality in ('audio', 'both'):
#             self.audio_encoder = AudioGlobalEncoder(
#                 input_dim=getattr(args, 'audio_dim', 1024),
#                 hidden_dim=args.hidden_dim
#             )

#         if self.modality == 'both':
#             self.audio_guided_attn = AudioGuidedAttention(
#                 visual_dim=args.low_dim,
#                 audio_dim=args.hidden_dim,
#                 hidden_dim=args.hidden_dim
#             )
            
#             # 投影音频引导的视觉特征
#             self.guided_proj = nn.Sequential(
#                 nn.Linear(args.hidden_dim, self.single_branch_dim),
#                 nn.LayerNorm(self.single_branch_dim),
#                 nn.ReLU(inplace=False)
#             )

#             # [新增] 投影音频纯内容特征
#             self.audio_proj = nn.Sequential(
#                 nn.Linear(args.hidden_dim, self.single_branch_dim),
#                 nn.LayerNorm(self.single_branch_dim),
#                 nn.ReLU(inplace=False)
#             )

#             # [修改] 门控网络：输入现在是 3 倍维度 (视觉 + 引导特征 + 音频纯内容)
#             self.audio_gate = nn.Sequential(
#                 nn.Linear(self.single_branch_dim * 3, self.single_branch_dim),
#                 nn.Sigmoid()
#             )
            
#             # 极低初始偏置，防初期扰乱
#             nn.init.constant_(self.audio_gate[0].bias, -5.0)
            
#             # 通道级 ReZero 参数
#             self.audio_alpha = nn.Parameter(torch.zeros(self.single_branch_dim))
            
#             # 融合 Dropout
#             self.fusion_dropout = nn.Dropout(0.5)

#             # [新增] 音频辅助分类器 (防梯度饿死)
#             self.audio_aux_classifier = nn.Linear(args.hidden_dim, args.num_classes)

#             # 最终分类器
#             self.final_classifier = nn.Sequential(
#                 nn.Dropout(0.3),
#                 nn.Linear(self.single_branch_dim, args.num_classes)
#             )
            
#         elif self.modality == 'visual':
#             self.final_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
#         elif self.modality == 'audio':
#             self.final_classifier = nn.Sequential(
#                 nn.Dropout(0.3),
#                 nn.Linear(args.hidden_dim, args.num_classes)
#             )
#         else:
#             raise ValueError(f"未知的 modality: {self.modality}")

#     @torch.no_grad()
#     def _update_prototypes_single(self, prototypes, embeddings_list,
#                                   bag_labels, instance_predictions_list):
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
#                     prototypes[pred_label] = (prototypes[pred_label] * self.args.proto_m
#                                               + (1 - self.args.proto_m) * emb)
#         prototypes.copy_(F.normalize(prototypes, p=2, dim=1))

#     def _hierarchical_aggregate(self, features_list):
#         prototypes = self.visual_prototypes
#         num_instances_per_bag = [f.shape[0] for f in features_list]
#         features_flat = torch.cat(features_list, dim=0)

#         instance_logits_flat, instance_embeddings_flat = self.visual_instance_classifier(features_flat)
#         sim_to_protos_flat = torch.matmul(instance_embeddings_flat, prototypes.t())

#         instance_logits_list = torch.split(instance_logits_flat, num_instances_per_bag, dim=0)
#         instance_embeddings_list = torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0)
#         sim_to_protos_list = torch.split(sim_to_protos_flat, num_instances_per_bag, dim=0)

#         final_bag_features = []
#         for i in range(len(instance_embeddings_list)):
#             embeddings = instance_embeddings_list[i]
#             sims = sim_to_protos_list[i]
#             cluster_assignments = torch.argmax(sims, dim=1)

#             cluster_features = []
#             for k in range(self.args.num_classes):
#                 indices_in_cluster = (cluster_assignments == k).nonzero(as_tuple=True)[0]
#                 if len(indices_in_cluster) == 0:
#                     cluster_feature = torch.zeros(self.args.low_dim, device=embeddings.device)
#                 else:
#                     cluster_feature, _ = torch.max(embeddings[indices_in_cluster], dim=0)
#                 cluster_features.append(cluster_feature)
#             final_bag_features.append(torch.cat(cluster_features, dim=0))

#         bag_features = torch.stack(final_bag_features, dim=0)
#         return bag_features, list(instance_logits_list), list(instance_embeddings_list)

#     def _prepare_audio(self, audio_features_list):
#         if isinstance(audio_features_list, list):
#             return torch.stack(
#                 [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list], dim=0)
#         return audio_features_list

#     def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
#         if self.modality == 'visual':
#             v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)
#             if self.training and bag_labels is not None:
#                 with torch.no_grad():
#                     v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
#                     self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
#             logits = self.final_classifier(v_bag)
#             return {
#                 'logits': logits,


#             }

#         elif self.modality == 'audio':
#             audio_tensor = self._prepare_audio(audio_features_list)
#             a_bag = self.audio_encoder(audio_tensor)
#             logits = self.final_classifier(a_bag)
#             return {
#                 'logits': logits,
       
#             }

#         elif self.modality == 'both':
#             # 1. 视觉主路径
#             v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)

#             if self.training and bag_labels is not None:
#                 with torch.no_grad():
#                     v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
#                     self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

#             # 2. 音频特征提取
#             audio_tensor = self._prepare_audio(audio_features_list)
#             a_global = self.audio_encoder(audio_tensor)

#             # 3. 注意力引导
#             B = len(visual_features_list)
#             guided_v_list = []
#             attn_weights_list = []

#             for i in range(B):
#                 v_inst = v_inst_embeds[i]           
#                 a_feat = a_global[i].unsqueeze(0)   
#                 guided_v, attn_w = self.audio_guided_attn(v_inst, a_feat)
#                 guided_v_list.append(guided_v)       
#                 attn_weights_list.append(attn_w)

#             guided_v_tensor = torch.cat(guided_v_list, dim=0) 
#             guided_v_proj = self.guided_proj(guided_v_tensor) 
            
#             # [新增] 提取音频纯内容特征
#             a_proj = self.audio_proj(a_global)

#             # 4. 终极门控融合
#             # 拼接: [视觉特征(阻断梯度), 音频引导的视觉特征, 音频纯内容特征]
#             gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
#             gate = self.audio_gate(gate_input) 

#             # 要注入的信息 = 音频上下文(引导) + 音频本体信息(内容)
#             combined_audio_info = guided_v_proj + a_proj 
            
#             # 计算音频残差
#             audio_residual = self.audio_alpha * gate * self.fusion_dropout(combined_audio_info)

#             # [新增] Audio Drop: 训练时 20% 概率强行阻断音频
#             if self.training and torch.rand(1).item() < 0.2:
#                 audio_residual = torch.zeros_like(audio_residual)

#             # 相加融合
#             fused_bag = v_bag + audio_residual

#             # 5. 分类输出
#             logits = self.final_classifier(fused_bag)
            

#             return {
#                 'logits': logits,
#                 'audio_aux_logits': audio_aux_logits,           # 用于算辅助 Loss
#                 'cross_attention_weights': attn_weights_list,
#                 'v_bag_detached': v_bag.detach(),               # 用于算正交 Loss
#                 'guided_v_proj': guided_v_proj                  # 用于算正交 Loss
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
    """跨模态注意力：用全局音频特征引导视觉序列 (带有温度锐化)"""
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_proj = nn.Linear(visual_dim, hidden_dim)
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        
        # [优化点] 注意力温度锐化，迫使注意力更集中
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, visual_instances, audio_global):
        Q = self.q_proj(audio_global)      # [1, hidden_dim]
        K = self.k_proj(visual_instances)  # [N, hidden_dim]
        V = self.v_proj(visual_instances)  # [N, hidden_dim]

        temp = torch.clamp(self.attn_temp, min=0.01)
        
        attn_scores = (torch.matmul(Q, K.t()) / self.scale) / temp  # [1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)               # [1, N]

        guided_visual_bag = torch.matmul(attn_weights, V)  # [1, hidden_dim]
        return guided_visual_bag, attn_weights


class LieDetection(nn.Module):
    """
    非对称多模态谎言检测网络 - 终极安全融合版
    包含: 内容直连、辅助分类器、通道级 ReZero、温度锐化、梯度截断、Audio Drop
    """
    def __init__(self, args, initial_prototypes=None):
        super().__init__()
        self.args = args
        self.modality = getattr(args, 'modality', 'both')
        self.single_branch_dim = args.low_dim * args.num_classes  # K * D

        if self.modality in ('visual', 'both'):
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
            self.audio_encoder = AudioGlobalEncoder(
                input_dim=getattr(args, 'audio_dim', 1024),
                hidden_dim=args.hidden_dim
            )

        if self.modality == 'both':
            self.audio_guided_attn = AudioGuidedAttention(
                visual_dim=args.low_dim,
                audio_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim
            )
            
            # 投影音频引导的视觉特征
            self.guided_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False)
            )

            # [新增] 投影音频纯内容特征
            self.audio_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False)
            )

            # [修改] 门控网络：输入现在是 3 倍维度 (视觉 + 引导特征 + 音频纯内容)
            self.audio_gate = nn.Sequential(
                nn.Linear(self.single_branch_dim * 3, self.single_branch_dim),
                nn.Sigmoid()
            )
            
            # 极低初始偏置，防初期扰乱
            nn.init.constant_(self.audio_gate[0].bias, -5.0)
            
            # 通道级 ReZero 参数
            self.audio_alpha = nn.Parameter(torch.zeros(self.single_branch_dim))
            
            # 融合 Dropout
            self.fusion_dropout = nn.Dropout(0.5)

            # 最终分类器
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim, args.num_classes)
            )
            
        elif self.modality == 'visual':
            self.final_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
        elif self.modality == 'audio':
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(args.hidden_dim, args.num_classes)
            )
        else:
            raise ValueError(f"未知的 modality: {self.modality}")

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
        if self.modality == 'visual':
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
            logits = self.final_classifier(v_bag)
            return {
                'logits': logits,


            }

        elif self.modality == 'audio':
            audio_tensor = self._prepare_audio(audio_features_list)
            a_bag = self.audio_encoder(audio_tensor)
            logits = self.final_classifier(a_bag)
            return {
                'logits': logits,
       
            }

        elif self.modality == 'both':
            # 1. 视觉主路径
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)

            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)

            # 2. 音频特征提取
            audio_tensor = self._prepare_audio(audio_features_list)
            a_global = self.audio_encoder(audio_tensor)

            # 3. 注意力引导
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
            
            # [新增] 提取音频纯内容特征
            a_proj = self.audio_proj(a_global)

            # 4. 终极门控融合
            # 拼接: [视觉特征(阻断梯度), 音频引导的视觉特征, 音频纯内容特征]
            gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
            gate = self.audio_gate(gate_input) 

            # 要注入的信息 = 音频上下文(引导) + 音频本体信息(内容)
            combined_audio_info = guided_v_proj + a_proj 
            
            # 计算音频残差
            audio_residual = self.audio_alpha * gate * self.fusion_dropout(combined_audio_info)

            # [新增] Audio Drop: 训练时 20% 概率强行阻断音频
            if self.training and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)

            # 相加融合
            fused_bag = v_bag + audio_residual

            # 5. 分类输出
            logits = self.final_classifier(fused_bag)
            

            return {
                'logits': logits,
                'cross_attention_weights': attn_weights_list,
                'v_bag_detached': v_bag.detach(),               # 用于算正交 Loss
                'guided_v_proj': guided_v_proj                  # 用于算正交 Loss
            }

