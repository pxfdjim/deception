import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualInstanceClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, embedding_dim=64, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        embedding = F.normalize(self.projector(features), p=2, dim=1)
        return logits, embedding


class AudioGlobalEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.network(x)


class AudioGuidedAttention(nn.Module):
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_proj = nn.Linear(visual_dim, hidden_dim)
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

    def forward(self, visual_instances, audio_global):
        query = self.q_proj(audio_global)
        key = self.k_proj(visual_instances)
        value = self.v_proj(visual_instances)
        temp = torch.clamp(self.attn_temp, min=0.01)
        attn_scores = (torch.matmul(query, key.t()) / self.scale) / temp
        attn_weights = F.softmax(attn_scores, dim=-1)
        guided_visual_bag = torch.matmul(attn_weights, value)
        return guided_visual_bag, attn_weights


class LieDetection(nn.Module):
    def __init__(self, args, initial_prototypes=None):
        super().__init__()
        self.args = args
        self.modality = getattr(args, "modality", "both")
        self.single_branch_dim = args.low_dim * args.num_classes

        self.use_topk_proto_update = getattr(args, "use_topk_proto_update", False)
        self.use_conservative_topk_proto_update = getattr(
            args, "use_conservative_topk_proto_update", False
        )
        self.use_cluster_topk_mean_pooling = getattr(args, "use_cluster_topk_mean_pooling", False)

        if self.modality in ("visual", "both"):
            self.visual_instance_classifier = VisualInstanceClassifier(
                input_dim=getattr(args, "visual_dim", 768),
                hidden_dim=args.hidden_dim,
                embedding_dim=args.low_dim,
                num_classes=args.num_classes,
            )
            if initial_prototypes is not None:
                self.register_buffer("visual_prototypes", initial_prototypes.clone())
            else:
                self.register_buffer("visual_prototypes", torch.zeros(args.num_classes, args.low_dim))

        if self.modality in ("audio", "both"):
            self.audio_encoder = AudioGlobalEncoder(
                input_dim=getattr(args, "audio_dim", 1024),
                hidden_dim=args.hidden_dim,
            )

        if self.modality == "both":
            self.audio_guided_attn = AudioGuidedAttention(
                visual_dim=args.low_dim,
                audio_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
            )
            self.guided_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False),
            )
            self.audio_proj = nn.Sequential(
                nn.Linear(args.hidden_dim, self.single_branch_dim),
                nn.LayerNorm(self.single_branch_dim),
                nn.ReLU(inplace=False),
            )
            self.audio_gate = nn.Sequential(
                nn.Linear(self.single_branch_dim * 3, self.single_branch_dim),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.audio_gate[0].bias, -5.0)
            self.fusion_dropout = nn.Dropout(0.5)
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.single_branch_dim, args.num_classes),
            )
        elif self.modality == "visual":
            self.final_classifier = nn.Linear(self.single_branch_dim, args.num_classes)
        elif self.modality == "audio":
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(args.hidden_dim, args.num_classes),
            )
        else:
            raise ValueError(f"未知的 modality: {self.modality}")

    def _topk_proto_active(self):
        if not self.use_topk_proto_update:
            return False
        current_epoch = getattr(self, "current_epoch", 0)
        warmup_epochs = getattr(self.args, "topk_proto_warmup_epochs", 0)
        return current_epoch >= warmup_epochs

    @torch.no_grad()
    def _update_prototypes_single(self, prototypes, embeddings_list, bag_labels, instance_logits_list):
        topk_active = self._topk_proto_active()
        for embeddings, bag_label, logits in zip(embeddings_list, bag_labels, instance_logits_list):
            label = int(bag_label.item())
            if label == 0:
                selected_embeddings = embeddings
            elif self.use_conservative_topk_proto_update and self.use_topk_proto_update and not topk_active:
                continue
            elif topk_active:
                lie_probs = F.softmax(logits.detach(), dim=-1)[:, 1]
                topk_ratio = getattr(self.args, "topk_proto_ratio", 0.25)
                threshold = getattr(self.args, "topk_proto_threshold", 0.0)
                num_top = max(1, int(round(embeddings.size(0) * topk_ratio)))
                top_indices = torch.topk(lie_probs, k=min(num_top, embeddings.size(0))).indices
                if threshold > 0:
                    top_indices = top_indices[lie_probs[top_indices] >= threshold]
                if top_indices.numel() == 0:
                    continue
                selected_embeddings = embeddings[top_indices]
            else:
                preds = torch.argmax(logits.detach(), dim=-1)
                indices = (preds == label).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                selected_embeddings = embeddings[indices]

            center = F.normalize(selected_embeddings.mean(dim=0), p=2, dim=0)
            momentum = getattr(self.args, "proto_m", 0.9)
            prototypes[label] = F.normalize(
                prototypes[label] * momentum + center * (1 - momentum),
                p=2,
                dim=0,
            )
        prototypes.copy_(F.normalize(prototypes, p=2, dim=1))

    def _pool_cluster_features(self, embeddings, sims, indices_in_cluster, class_idx):
        if indices_in_cluster.numel() == 0:
            return torch.zeros(self.args.low_dim, device=embeddings.device)

        cluster_embeddings = embeddings[indices_in_cluster]
        if not self.use_cluster_topk_mean_pooling:
            cluster_feature, _ = torch.max(cluster_embeddings, dim=0)
            return cluster_feature

        cluster_sims = sims[indices_in_cluster, class_idx].detach()
        topk_ratio = getattr(self.args, "cluster_topk_mean_ratio", 0.5)
        num_top = max(1, int(round(cluster_embeddings.size(0) * topk_ratio)))
        num_top = min(num_top, cluster_embeddings.size(0))
        top_indices = torch.topk(cluster_sims, k=num_top, largest=True).indices
        return cluster_embeddings[top_indices].mean(dim=0)

    def _hierarchical_aggregate(self, features_list):
        num_instances_per_bag = [f.shape[0] for f in features_list]
        features_flat = torch.cat(features_list, dim=0)

        instance_logits_flat, instance_embeddings_flat = self.visual_instance_classifier(features_flat)
        sim_to_protos_flat = torch.matmul(instance_embeddings_flat, self.visual_prototypes.t())

        instance_logits_list = torch.split(instance_logits_flat, num_instances_per_bag, dim=0)
        instance_embeddings_list = torch.split(instance_embeddings_flat, num_instances_per_bag, dim=0)
        sim_to_protos_list = torch.split(sim_to_protos_flat, num_instances_per_bag, dim=0)

        final_bag_features = []
        for embeddings, sims in zip(instance_embeddings_list, sim_to_protos_list):
            cluster_assignments = torch.argmax(sims, dim=1)
            cluster_features = []
            for class_idx in range(self.args.num_classes):
                indices = (cluster_assignments == class_idx).nonzero(as_tuple=True)[0]
                cluster_features.append(
                    self._pool_cluster_features(embeddings, sims, indices, class_idx)
                )
            final_bag_features.append(torch.cat(cluster_features, dim=0))

        bag_features = torch.stack(final_bag_features, dim=0)
        return bag_features, list(instance_logits_list), list(instance_embeddings_list)

    def _prepare_audio(self, audio_features_list):
        if isinstance(audio_features_list, list):
            return torch.stack(
                [a.squeeze(0) if a.dim() == 2 else a for a in audio_features_list],
                dim=0,
            )
        return audio_features_list

    def forward(self, visual_features_list=None, audio_features_list=None, bag_labels=None):
        if self.modality == "audio":
            audio_tensor = self._prepare_audio(audio_features_list)
            a_bag = self.audio_encoder(audio_tensor)
            return {"logits": self.final_classifier(a_bag)}

        v_bag, v_inst_logits, v_inst_embeds = self._hierarchical_aggregate(visual_features_list)
        if self.training and bag_labels is not None:
            with torch.no_grad():
                self._update_prototypes_single(
                    self.visual_prototypes,
                    v_inst_embeds,
                    bag_labels,
                    v_inst_logits,
                )

        if self.modality == "visual":
            logits = self.final_classifier(v_bag)
            return {
                "logits": logits,
                "instance_logits_list": v_inst_logits,
            }

        audio_tensor = self._prepare_audio(audio_features_list)
        a_global = self.audio_encoder(audio_tensor)

        guided_v_list = []
        attn_weights_list = []
        for i, v_inst in enumerate(v_inst_embeds):
            guided_v, attn_w = self.audio_guided_attn(v_inst, a_global[i].unsqueeze(0))
            guided_v_list.append(guided_v)
            attn_weights_list.append(attn_w)

        guided_v_proj = self.guided_proj(torch.cat(guided_v_list, dim=0))
        a_proj = self.audio_proj(a_global)
        gate_input = torch.cat([v_bag.detach(), guided_v_proj, a_proj], dim=-1)
        gate = self.audio_gate(gate_input)

        audio_residual = gate * self.fusion_dropout(guided_v_proj + a_proj)
        fused_bag = v_bag + audio_residual
        logits = self.final_classifier(fused_bag)

        v_feat_norm = F.normalize(v_bag.detach(), p=2, dim=1)
        a_feat_norm = F.normalize(guided_v_proj, p=2, dim=1)
        ortho_loss = torch.abs(torch.sum(v_feat_norm * a_feat_norm, dim=1)).mean()

        return {
            "logits": logits,
            "instance_logits_list": v_inst_logits,
            "cross_attention_weights": attn_weights_list,
            "ortho_loss": ortho_loss,
        }
