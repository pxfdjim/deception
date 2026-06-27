# import os

# class Args:
#     """参数配置类 - DOLOS数据集"""
#     def __init__(self):
#         # DOLOS数据相关
#         self.feature_root = "/home/pengxf/emotion/dataset/DOLOS/features/videomaev2"
#         self.fold_path = "/home/pengxf/emotion/dataset/DOLOS/Training_Protocols"
#         self.fold_index = 0  # 当前使用的fold索引 (0-2)
#         # 确保在args中添加新参数:
        
#         # ====== 模态设置 ======
#         # 可选: 'visual' (纯视觉), 'audio' (纯音频), 'both' (视觉+音频 Late Fusion)
#         self.modality = 'audio'
#         self.visual_dim = 768     # VideoMAE v2 特征维度
#         self.audio_dim = 1024     # WavLM Large 特征维度
#         self.audio_feature_root = "/home/pengxf/emotion/dataset/DOLOS/wavlm_large_features"
        
#         self.lambda_main = 1.5
#         # 模型参数
#         self.input_dim = self.visual_dim  # 向后兼容（其他模型变体使用）
#         self.hidden_dim = 768     # 隐藏层维度
#         self.low_dim = 128        # 对比学习特征维度
#         self.num_classes = 2      # 二分类：真话/谎言
#         self.momentum = 0.9


#         self.proto_m = 0.9        # 原型更新动量
#         self.partial_rate = 0.1   # 部分标签比例

        
#         # 训练参数
#         self.epochs = 100
#         self.batch_size = 8       # 小批次适应数据集大小
#         self.lr = 0.001           # 学习率
#         self.weight_decay = 5e-6

#         self.aggr_method = 'hierarchical_max'  # 层次聚合方法: 'hierarchical_mean', 'mean', 'max'
#         # 学习率调度
#         self.lr_decay_epochs = [30, 40]
#         self.lr_decay_rate = 0.3
#         self.cosine = False
#         self.ortho_weight = 0.1
#         # 其他
#         self.print_freq = 10
#         self.num_workers = 1
#         self.seed = 42
#         self.num_runs = 3  # DOLOS使用3折交叉验证
#         self.optimizer = "sgd"
#         self.warmup_epochs = 5
#         # self.align_weight = 0.05   # 模态对齐损失权重 (辅助正则，非强约束)
#         # 保存路径
#         self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/exper_model/dolo_audio"
#         os.makedirs(self.exp_dir, exist_ok=True)
        
#         # 消融实验配置（默认为False，即不消融）
#         self.ablate_hierarchical = False      # 消融层次聚合
#         self.ablate_attention = False         # 消融跨模态注意力
#         self.ablate_audio_content = False     # 消融音频纯内容
#         self.ablate_prototype_update = False  # 消融原型更新
#         self.ablate_gate = False              # 消融门控机制
#         self.ablate_rezero = False            # 消融ReZero参数
#         self.ablate_audio_drop = False        # 消融Audio Drop
#         self.ablate_temperature = False       # 消融注意力温度
#         self.ablate_layernorm = False         # 消融LayerNorm
#         self.ablate_fusion_dropout = False    # 消融融合Dropout
#         self.ablate_gradient_stop = False     # 消融梯度阻断
#         self.ablate_projection = False        # 消融特征投影
#         self.ablate_fusion_method = 'add'     # 融合方式: 'add'(相加), 'concat'(拼接), 'multiply'(乘法)




        
import os

class Args:
    """参数配置类 - DOLOS数据集"""
    def __init__(self):
        # DOLOS数据相关
        self.feature_root = "/home/pengxf/emotion/dataset/DOLOS/features/videomaev2"
        self.fold_path = "/home/pengxf/emotion/dataset/DOLOS/Training_Protocols"
        self.fold_index = 0  # 当前使用的fold索引 (0-2)
        # 确保在args中添加新参数:
        
        # ====== 模态设置 ======
        # 可选: 'visual' (纯视觉), 'audio' (纯音频), 'both' (视觉+音频 Late Fusion)
        self.modality = 'both'
        self.visual_dim = 768     # VideoMAE v2 特征维度
        self.audio_dim = 1024     # WavLM Large 特征维度
        self.audio_feature_root = "/home/pengxf/emotion/dataset/DOLOS/wavlm_large_features"
        
        self.lambda_main = 1.5
        # 模型参数
        self.input_dim = self.visual_dim  # 向后兼容（其他模型变体使用）
        self.hidden_dim = 768     # 隐藏层维度
        self.low_dim = 128        # 对比学习特征维度
        self.num_classes = 2      # 二分类：真话/谎言
        self.use_instance_loss = True
        self.instance_loss_weight = 0.1
        self.positive_instance_topk_ratio = 0.25
        self.use_topk_proto_update = False
        self.use_conservative_topk_proto_update = False
        self.topk_proto_ratio = 0.25
        self.topk_proto_threshold = 0.6
        self.topk_proto_warmup_epochs = 10
        self.proto_sep_margin = 0.2
        self.proto_sep_loss_weight = 0.05
        self.use_cluster_topk_mean_pooling = False
        self.cluster_topk_mean_ratio = 0.5
        self.use_visual_logit_ensemble = False
        self.visual_logit_ensemble_weight = 0.3
        self.use_audio_residual_drop = False
        self.audio_residual_drop_prob = 0.3
        self.use_visual_aux_loss = False
        self.visual_aux_loss_weight = 0.2
        self.visual_aux_loss_warmup_epochs = 0
        self.use_fusion_consistency_loss = False
        self.fusion_consistency_loss_weight = 0.05
        self.fusion_consistency_temperature = 2.0
        self.fusion_consistency_warmup_epochs = 0
        self.use_mil_evidence_loss = False
        self.mil_evidence_loss_weight = 0.1
        self.mil_evidence_topk_ratio = 0.25
        self.mil_evidence_rank_weight = 0.05
        self.mil_evidence_rank_margin = 0.5
        self.mil_evidence_warmup_epochs = 0
        self.use_eval_threshold_search = False
        self.eval_threshold_min = 0.2
        self.eval_threshold_max = 0.8
        self.eval_threshold_step = 0.01
        self.momentum = 0.9


        self.proto_m = 0.9        # 原型更新动量
        self.partial_rate = 0.1   # 部分标签比例

        
        # 训练参数
        self.epochs = 100
        self.batch_size = 8       # 小批次适应数据集大小
        self.lr = 0.001           # 学习率
        self.weight_decay = 5e-6
        self.label_smoothing = 0.0
        self.lie_class_weight = 1.5
        self.use_logit_margin_regularization = False
        self.logit_margin_weight = 0.02
        self.logit_margin_target = 3.0
        self.logit_margin_warmup_epochs = 10
        self.use_batch_rank_loss = False
        self.batch_rank_loss_weight = 0.05
        self.batch_rank_margin = 0.5
        self.batch_rank_warmup_epochs = 5
        self.use_model_ema = False
        self.model_ema_decay = 0.995
        self.model_ema_start_epoch = 10
        self.use_ema_dual_eval = False

        self.aggr_method = 'hierarchical_max'  # 层次聚合方法: 'hierarchical_mean', 'mean', 'max'
        # 学习率调度
        self.lr_decay_epochs = [30, 40]
        self.lr_decay_rate = 0.3
        self.cosine = False
        self.ortho_weight = 0.1
        # 其他
        self.print_freq = 10
        self.num_workers = 1
        self.seed = 42
        self.num_runs = 3  # DOLOS使用3折交叉验证
        self.optimizer = "sgd"
        self.warmup_epochs = 5
        # self.align_weight = 0.05   # 模态对齐损失权重 (辅助正则，非强约束)
        # 保存路径
        self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/photo"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 消融实验融合模式 (仅用于消融实验)
        # 可选: 'full', 'only_visual', 'only_audio', 'no_attention', 
        #       'no_gate', 'no_rezero', 'no_audio_content', 'no_audio_drop',
        #       'early_fusion', 'late_fusion'
        self.fusion_mode = 'only_visual'




        
