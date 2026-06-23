# import torch.nn as nn
# import torch.nn.functional as F
# import torch

# # InstanceClassifier 保持不变

# class InstanceClassifier(nn.Module):
#     """实例分类器模块 (保持不变)"""
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
# class LieDetection(nn.Module):
#     """
#     == 结合版：层次化聚合 (主干) + KL一致性损失 (辅助) ==
#     """
#     def __init__(self, args, initial_prototypes=None): # (我假设您已添加了初始化功能)
#         super().__init__()
#         self.args = args
#         self.instance_classifier = InstanceClassifier(
#             input_dim=args.input_dim, hidden_dim=args.hidden_dim,
#             embedding_dim=args.low_dim, num_classes=args.num_classes
#         )
#         self.bag_constraint_mlp = nn.Sequential(
#             nn.Linear(args.low_dim, 32),
#             nn.ReLU(inplace=False),
#             nn.Linear(32, args.num_classes)
#         )
        
#         # 保持模型二（层次化）的分类器
#         fused_feature_dim = args.low_dim * args.num_classes
#         self.final_classifier = nn.Linear(fused_feature_dim, args.num_classes)

#         # 注册原型 (假设已支持初始化)
#         if initial_prototypes is not None:
#             self.register_buffer("prototypes", initial_prototypes.clone())
#         else:
#             self.register_buffer("prototypes", torch.zeros(args.num_classes, args.low_dim))

#     @torch.no_grad()
#     def _update_prototypes(self, embeddings_list, bag_labels, instance_predictions_list):
#         # ... (此函数保持不变) ...
#         for i in range(len(embeddings_list)):
#             bag_label = bag_labels[i]
#             bag_embeddings = embeddings_list[i]
#             if bag_label == 0:
#                 for emb in bag_embeddings:
#                     self.prototypes[0] = self.prototypes[0] * self.args.proto_m + (1 - self.args.proto_m) * emb
#             else:
#                 preds = instance_predictions_list[i]
#                 for j, emb in enumerate(bag_embeddings):
#                     pred_label = preds[j]
#                     self.prototypes[pred_label] = self.prototypes[pred_label] * self.args.proto_m + (1 - self.args.proto_m) * emb
#         self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

#     def forward(self, features_list, bag_labels):
#         num_instances_per_bag = [f.shape[0] for f in features_list]
#         features_flat = torch.cat(features_list, dim=0)
        
#         # 1. 实例级别的所有输出 (两个模型都需要)
#         instance_logits_flat, instance_embeddings_flat = self.instance_classifier(features_flat)
#         instance_logits_list = list(torch.split(instance_logits_flat, num_instances_per_bag, dim=0))
#         instance_embeddings_list = list(torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0))
        
        
#         # =================================================================
#         # 2. (新增) 来自模型一的 KL 损失，作为辅助正则化项
#         # =================================================================
#         # (确保原型不为零，如果使用了初始化，这步会更安全)
#         safe_prototypes = self.prototypes.clone().detach() 
#         sim_to_protos_flat = torch.matmul(instance_embeddings_flat, safe_prototypes.t())
#         dist_p = F.log_softmax(instance_logits_flat, dim=-1)
#         dist_q = F.softmax(sim_to_protos_flat, dim=-1)
#         kl_loss = F.kl_div(dist_p, dist_q, reduction='batchmean')
        
        
#         # =================================================================
#         # 3. (主干) 来自模型二的层次化聚合
#         # =================================================================
#         final_bag_features = []
#         for i in range(len(instance_embeddings_list)):
#             embeddings = instance_embeddings_list[i] # [N, D]
            
#             # 聚类: 使用原型将实例分配到 K 个簇
#             # (注意：我们这里用已经计算好的 flat 结果来切片，避免重复计算)
#             sim_to_protos = sim_to_protos_flat[sum(num_instances_per_bag[:i]):sum(num_instances_per_bag[:i+1])]
#             cluster_assignments = torch.argmax(sim_to_protos, dim=1) # [N]

#             # 簇内聚合 (Tier-1)
#             cluster_features = []
#             for k in range(self.args.num_classes):
#                 indices_in_cluster = (cluster_assignments == k).nonzero(as_tuple=True)[0]
                
#                 if len(indices_in_cluster) == 0:
#                     cluster_feature = torch.zeros(self.args.low_dim, device=embeddings.device)
#                     # cluster_feature = self.prototypes[k].clone().detach()
#                 else:
#                     embeddings_in_cluster = embeddings[indices_in_cluster]
#                     cluster_feature, _ = torch.max(embeddings_in_cluster, dim=0)
#                 cluster_features.append(cluster_feature)
            
#             # 簇间聚合 (Tier-2)
#             final_bag_feature = torch.cat(cluster_features, dim=0) # [K*D]
#             final_bag_features.append(final_bag_feature)
            
#         final_bag_features_tensor = torch.stack(final_bag_features, dim=0)
        
#         # 最终分类
#         logits = self.final_classifier(final_bag_features_tensor)
        
#         # 5. (不变) 原型更新
#         if self.training and bag_labels is not None:
#             with torch.no_grad():
#                 instance_predictions_list = [torch.argmax(logits.detach(), dim=-1) for logits in instance_logits_list] 
#             self._update_prototypes(instance_embeddings_list, bag_labels, instance_predictions_list)

#         # 6. 返回所有结果
#         return {
#             'instance_logits_list': instance_logits_list,
#             'instance_embeddings_list': instance_embeddings_list,
#             'logits':logits,                 # <<< 来自模型二 (层次化聚合)
#             'kl_loss': kl_loss,                       # <<< 来自模型一 (一致性损失)
#             # ... 保持API兼容性 ...
#             'attention_weights': [None for _ in features_list],
#         }


import torch.nn as nn
import torch.nn.functional as F
import torch

class InstanceClassifier(nn.Module):
    """实例分类器模块 (保持不变)"""
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

class LieDetection(nn.Module):
    """
    原型引导层次化聚合 (PPLG) 谎言检测模型
    支持三种模态模式:
      - 'visual':  仅视觉 (向后兼容原始接口)
      - 'audio':   仅音频
      - 'both':    Late Fusion — 两个独立分支 + 包级拼接分类
    """
    def __init__(self, args, initial_prototypes=None):
        super().__init__()
        self.args = args
        self.modality = getattr(args, 'modality', 'visual')
        
        single_branch_dim = args.low_dim * args.num_classes  # K * D
        
        if self.modality == 'visual':
            # ─── 纯视觉模态 (完全向后兼容) ───
            self.instance_classifier = InstanceClassifier(
                input_dim=getattr(args, 'visual_dim', args.input_dim),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim, num_classes=args.num_classes
            )
            if initial_prototypes is not None:
                self.register_buffer("prototypes", initial_prototypes.clone())
            else:
                self.register_buffer("prototypes", torch.zeros(args.num_classes, args.low_dim))
            fused_dim = single_branch_dim
            
        elif self.modality == 'audio':
            # ─── 纯音频模态 ───
            self.instance_classifier = InstanceClassifier(
                input_dim=getattr(args, 'audio_dim', 1024),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim, num_classes=args.num_classes
            )
            self.register_buffer("prototypes", torch.zeros(args.num_classes, args.low_dim))
            fused_dim = single_branch_dim
            
        elif self.modality == 'both':
            # ─── 双模态 Late Fusion ───
            # 视觉分支
            self.visual_instance_classifier = InstanceClassifier(
                input_dim=getattr(args, 'visual_dim', 768),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim, num_classes=args.num_classes
            )
            self.register_buffer("visual_prototypes", torch.zeros(args.num_classes, args.low_dim))
            
            # 音频分支
            self.audio_instance_classifier = InstanceClassifier(
                input_dim=getattr(args, 'audio_dim', 1024),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim, num_classes=args.num_classes
            )
            self.register_buffer("audio_prototypes", torch.zeros(args.num_classes, args.low_dim))
            
            fused_dim = single_branch_dim * 2  # 视觉 + 音频 拼接
        else:
            raise ValueError(f"未知的 modality: {self.modality}")
        
        # 最终包级分类器
        self.final_classifier = nn.Linear(fused_dim, args.num_classes)

    # ─────────────────────────────────────────────────
    #  原型更新（单分支通用版）
    # ─────────────────────────────────────────────────
    @torch.no_grad()
    def _update_prototypes_single(self, prototypes, embeddings_list, 
                                   bag_labels, instance_predictions_list):
        """更新某一模态的原型"""
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
                    prototypes[pred_label] = prototypes[pred_label] * self.args.proto_m + (1 - self.args.proto_m) * emb
        prototypes.copy_(F.normalize(prototypes, p=2, dim=1))

    # ─────────────────────────────────────────────────
    #  单分支 层次化聚合
    # ─────────────────────────────────────────────────
    def _hierarchical_aggregate(self, features_list, instance_classifier, prototypes):
        """
        单个模态分支的处理流程:
          实例分类 → 原型相似度 → 聚类分配 → 簇内Max Pooling → 簇间拼接
        
        Args:
            features_list: list of [N_i, D_in] 每个包的实例特征
            instance_classifier: 该模态的InstanceClassifier
            prototypes: 该模态的原型 [K, low_dim]
        
        Returns:
            bag_features: [B, K * low_dim] 包级特征
            instance_logits_list: list of [N_i, num_classes]
            instance_embeddings_list: list of [N_i, low_dim]
        """
        num_instances_per_bag = [f.shape[0] for f in features_list]
        features_flat = torch.cat(features_list, dim=0)
        
        # 实例级分类 + 投影
        instance_logits_flat, instance_embeddings_flat = instance_classifier(features_flat)
        
        # 与原型计算相似度
        sim_to_protos_flat = torch.matmul(instance_embeddings_flat, prototypes.t())
        
        # 切分回各个包
        instance_logits_list = torch.split(instance_logits_flat, num_instances_per_bag, dim=0)
        instance_embeddings_list = torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0)
        sim_to_protos_list = torch.split(sim_to_protos_flat, num_instances_per_bag, dim=0)
        
        # 层次化聚合
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
                    embeddings_in_cluster = embeddings[indices_in_cluster]
                    cluster_feature, _ = torch.max(embeddings_in_cluster, dim=0)
                cluster_features.append(cluster_feature)
            
            final_bag_feature = torch.cat(cluster_features, dim=0)  # [K * low_dim]
            final_bag_features.append(final_bag_feature)
        
        bag_features = torch.stack(final_bag_features, dim=0)  # [B, K * low_dim]
        return bag_features, list(instance_logits_list), list(instance_embeddings_list)

    # ─────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────
    def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
        """
        Args:
            visual_features_list: list of [N_v_i, visual_dim] 或 None
            audio_features_list:  list of [N_a_i, audio_dim]  或 None
            bag_labels:           [B] 包级标签 (训练时提供，用于更新原型)
        
        Returns:
            dict: {
                'logits': [B, num_classes],
                'instance_logits_list': list of [N_i, num_classes],
                'instance_embeddings_list': list of [N_i, low_dim],
                'attention_weights': list of None (保持兼容),
            }
        """
        if self.modality == 'visual':
            # ─── 纯视觉 ───
            bag_features, inst_logits, inst_embeds = self._hierarchical_aggregate(
                visual_features_list, self.instance_classifier, self.prototypes)
            
            # 原型更新
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    inst_preds = [torch.argmax(l.detach(), dim=-1) for l in inst_logits]
                    self._update_prototypes_single(
                        self.prototypes, inst_embeds, bag_labels, inst_preds)
            
            logits = self.final_classifier(bag_features)
            return {
                'instance_logits_list': inst_logits,
                'instance_embeddings_list': inst_embeds,
                'logits': logits,
                'attention_weights': [None] * len(visual_features_list),
            }
        
        elif self.modality == 'audio':
            # ─── 纯音频 ───
            bag_features, inst_logits, inst_embeds = self._hierarchical_aggregate(
                audio_features_list, self.instance_classifier, self.prototypes)
            
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    inst_preds = [torch.argmax(l.detach(), dim=-1) for l in inst_logits]
                    self._update_prototypes_single(
                        self.prototypes, inst_embeds, bag_labels, inst_preds)
            
            logits = self.final_classifier(bag_features)
            return {
                'instance_logits_list': inst_logits,
                'instance_embeddings_list': inst_embeds,
                'logits': logits,
                'attention_weights': [None] * len(audio_features_list),
            }
        
        elif self.modality == 'both':
            # ─── Late Fusion: 双分支独立处理 → 包级拼接 ───
            # 视觉分支
            v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(
                visual_features_list, self.visual_instance_classifier, self.visual_prototypes)
            
            # 音频分支
            a_bag, a_inst_logits, a_inst_embeds = self._hierarchical_aggregate(
                audio_features_list, self.audio_instance_classifier, self.audio_prototypes)
            
            # 包级特征拼接 → 分类
            fused_bag = torch.cat([v_bag, a_bag], dim=1)  # [B, 2*K*low_dim]
            logits = self.final_classifier(fused_bag)
            
            # 合并两个模态的实例输出（供实例级损失使用）
            merged_inst_logits = [torch.cat([v, a], dim=0) 
                                  for v, a in zip(v_inst_logits, a_inst_logits)]
            merged_inst_embeds = [torch.cat([v, a], dim=0) 
                                  for v, a in zip(v_inst_embeds, a_inst_embeds)]
            
            # 分别更新各模态原型
            if self.training and bag_labels is not None:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(
                        self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
                    
                    a_preds = [torch.argmax(l.detach(), dim=-1) for l in a_inst_logits]
                    self._update_prototypes_single(
                        self.audio_prototypes, a_inst_embeds, bag_labels, a_preds)
            
            return {
                'instance_logits_list': merged_inst_logits,
                'instance_embeddings_list': merged_inst_embeds,
                'logits': logits,
                'attention_weights': [None] * len(visual_features_list),
            }

