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

    def forward(self, visual_instances, audio_global, ablate_temperature=False):
        Q = self.q_proj(audio_global)      # [1, hidden_dim]
        K = self.k_proj(visual_instances)  # [N, hidden_dim]
        V = self.v_proj(visual_instances)  # [N, hidden_dim]

        if ablate_temperature:
            # 消融版本：使用固定温度
            attn_scores = torch.matmul(Q, K.t()) / self.scale  # [1, N]
        else:
            # 完整版本：使用可学习温度
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
        
        # 消融实验配置
        self.ablate_hierarchical = getattr(args, 'ablate_hierarchical', False)  # 消融层次聚合
        self.ablate_attention = getattr(args, 'ablate_attention', False)  # 消融跨模态注意力
        self.ablate_audio_content = getattr(args, 'ablate_audio_content', False)  # 消融音频纯内容
        self.ablate_prototype_update = getattr(args, 'ablate_prototype_update', False)  # 消融原型更新
        self.ablate_gate = getattr(args, 'ablate_gate', False)  # 消融门控机制，使用直接融合
        self.ablate_rezero = getattr(args, 'ablate_rezero', False)  # 消融ReZero参数
        self.ablate_audio_drop = getattr(args, 'ablate_audio_drop', False)  # 消融Audio Drop
        self.ablate_temperature = getattr(args, 'ablate_temperature', False)  # 消融注意力温度
        self.ablate_layernorm = getattr(args, 'ablate_layernorm', False)  # 消融LayerNorm
        self.ablate_fusion_dropout = getattr(args, 'ablate_fusion_dropout', False)  # 消融融合Dropout
        self.ablate_gradient_stop = getattr(args, 'ablate_gradient_stop', False)  # 消融梯度阻断
        self.ablate_projection = getattr(args, 'ablate_projection', False)  # 消融特征投影
        self.ablate_fusion_method = getattr(args, 'ablate_fusion_method', 'add')  # 融合方式: add/concat/multiply

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
            if self.ablate_projection:
                # 消融版本：使用简单线性投影
                self.guided_proj = nn.Linear(args.hidden_dim, self.single_branch_dim)
            else:
                # 完整版本：使用复杂投影网络
                if self.ablate_layernorm:
                    # 消融LayerNorm版本
                    self.guided_proj = nn.Sequential(
                        nn.Linear(args.hidden_dim, self.single_branch_dim),
                        nn.ReLU(inplace=False)
                    )
                else:
                    # 完整版本
                    self.guided_proj = nn.Sequential(
                        nn.Linear(args.hidden_dim, self.single_branch_dim),
                        nn.LayerNorm(self.single_branch_dim),
                        nn.ReLU(inplace=False)
                    )

            # [新增] 投影音频纯内容特征
            if self.ablate_projection:
                # 消融版本：使用简单线性投影
                self.audio_proj = nn.Linear(args.hidden_dim, self.single_branch_dim)
            else:
                # 完整版本：使用复杂投影网络
                if self.ablate_layernorm:
                    # 消融LayerNorm版本
                    self.audio_proj = nn.Sequential(
                        nn.Linear(args.hidden_dim, self.single_branch_dim),
                        nn.ReLU(inplace=False)
                    )
                else:
                    # 完整版本
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
            if not self.ablate_fusion_dropout:
                self.fusion_dropout = nn.Dropout(0.5)
            else:
                self.fusion_dropout = nn.Identity()  # 消融版本：不使用Dropout

            # 最终分类器 - 根据融合方式选择输入维度
            if self.ablate_fusion_method == 'concat':
                # 拼接融合需要更大的分类器输入
                self.final_classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(self.single_branch_dim * 2, args.num_classes)  # 视觉+音频拼接
                )
            else:
                # 相加或乘法融合
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

        # 消融层次聚合：使用简单的mean pooling代替聚类聚合
        if self.ablate_hierarchical:
            print(f"[DEBUG] 使用消融版本：层次聚合 -> mean pooling")
            final_bag_features = []
            for i in range(len(instance_embeddings_list)):
                embeddings = instance_embeddings_list[i]
                # 简单平均，然后复制K次以匹配维度
                bag_feat = embeddings.mean(dim=0)  # [low_dim]
                bag_feat_repeated = bag_feat.repeat(self.args.num_classes)  # [K * low_dim]
                final_bag_features.append(bag_feat_repeated)
            bag_features = torch.stack(final_bag_features, dim=0)
        else:
            # 原始的层次聚合
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

            # 消融原型更新：跳过原型更新
            if self.training and bag_labels is not None and not self.ablate_prototype_update:
                with torch.no_grad():
                    v_preds = [torch.argmax(l.detach(), dim=-1) for l in v_inst_logits]
                    self._update_prototypes_single(self.visual_prototypes, v_inst_embeds, bag_labels, v_preds)
            elif self.ablate_prototype_update and self.training:
                print(f"[DEBUG] 使用消融版本：原型更新 -> 跳过更新")

            # 2. 音频特征提取
            audio_tensor = self._prepare_audio(audio_features_list)
            a_global = self.audio_encoder(audio_tensor)

            # 3. 注意力引导
            B = len(visual_features_list)
            guided_v_list = []
            attn_weights_list = []

            # 消融跨模态注意力：直接使用音频特征，不做引导
            if self.ablate_attention:
                print(f"[DEBUG] 使用消融版本：跨模态注意力 -> 直接使用音频特征")
                for i in range(B):
                    a_feat = a_global[i].unsqueeze(0)
                    guided_v_list.append(a_feat)
                    attn_weights_list.append(None)
            else:
                for i in range(B):
                    v_inst = v_inst_embeds[i]           
                    a_feat = a_global[i].unsqueeze(0)   
                    guided_v, attn_w = self.audio_guided_attn(v_inst, a_feat, self.ablate_temperature)
                    guided_v_list.append(guided_v)       
                    attn_weights_list.append(attn_w)

            guided_v_tensor = torch.cat(guided_v_list, dim=0) 
            guided_v_proj = self.guided_proj(guided_v_tensor) 
            
            # [新增] 提取音频纯内容特征
            a_proj = self.audio_proj(a_global)

            # 4. 终极门控融合
            # 消融音频纯内容：只使用引导特征，不使用音频内容
            if self.ablate_audio_content:
                print(f"[DEBUG] 使用消融版本：音频纯内容 -> 只使用引导特征")
                combined_audio_info = guided_v_proj
                if self.ablate_gradient_stop:
                    gate_input = torch.cat([v_bag, guided_v_proj, torch.zeros_like(a_proj)], dim=-1)
                else:
                    gate_input = torch.cat([v_bag.detach(), guided_v_proj, torch.zeros_like(a_proj)], dim=-1)
            else:
                # 拼接: [视觉特征(可选阻断梯度), 音频引导的视觉特征, 音频纯内容特征]
                if self.ablate_gradient_stop:
                    gate_input = torch.cat([v_bag, guided_v_proj, a_proj], dim=-1)
                else:
                    gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
                # 要注入的信息 = 音频上下文(引导) + 音频本体信息(内容)
                combined_audio_info = guided_v_proj + a_proj
            
            gate = self.audio_gate(gate_input) 
            
            # 消融门控机制：直接融合 vs 门控融合
            if self.ablate_gate:
                print(f"[DEBUG] 使用消融版本：门控机制 -> 直接融合")
                # 直接融合：使用固定权重，不使用门控网络
                if self.ablate_rezero:
                    # 消融ReZero：使用固定权重1.0
                    audio_residual = self.fusion_dropout(combined_audio_info)
                else:
                    # 使用ReZero参数
                    audio_residual = self.audio_alpha * self.fusion_dropout(combined_audio_info)
            else:
                # 门控融合：使用门控网络动态调节融合权重
                if self.ablate_rezero:
                    # 消融ReZero：使用固定权重1.0
                    audio_residual = gate * self.fusion_dropout(combined_audio_info)
                else:
                    # 使用ReZero参数
                    audio_residual = self.audio_alpha * gate * self.fusion_dropout(combined_audio_info)

            # [新增] Audio Drop: 训练时 20% 概率强行阻断音频
            if self.training and not self.ablate_audio_drop and torch.rand(1).item() < 0.2:
                audio_residual = torch.zeros_like(audio_residual)

            # 融合策略消融
            if self.ablate_fusion_method == 'concat':
                # 拼接融合
                fused_bag = torch.cat([v_bag, audio_residual], dim=-1)
            elif self.ablate_fusion_method == 'multiply':
                # 乘法融合
                fused_bag = v_bag * (1 + audio_residual)
            else:
                # 相加融合 (默认)
                fused_bag = v_bag + audio_residual

            # 5. 分类输出
            logits = self.final_classifier(fused_bag)
            # 6. [新增] 在模型内部计算正交特征 Loss
            # ==========================================
            # 分别进行 L2 归一化
            v_feat_norm = F.normalize(v_bag.detach(), p=2, dim=1)
            a_feat_norm = F.normalize(guided_v_proj, p=2, dim=1)
            
            # 计算 Batch 内的余弦相似度绝对值的均值
            cos_sim = torch.abs(torch.sum(v_feat_norm * a_feat_norm, dim=1))
            ortho_loss = cos_sim.mean()

            return {
                'logits': logits,
                'cross_attention_weights': attn_weights_list,
                'ortho_loss': ortho_loss,         
            }