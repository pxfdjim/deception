import os

class Args:
    """NumberGuess 数据集参数配置类 - 5折交叉验证"""
    def __init__(self):
        # NumberGuess数据相关
        self.feature_root = "/home/pengxf/jim/work/GSR/features_new"
        self.fold_path = "/home/deception/dataset/number_guess_v2/fold_split_new.csv"
        self.include_baseline = False    # 是否包含baseline问题
        self.target_only = True          # 是否只使用target问题
        
        # ====== 模态设置 ======
        # 可选: 'visual' (纯视觉), 'audio' (纯音频), 'both' (视觉+音频 Late Fusion)
        self.modality = 'audio'
        self.visual_dim = 768     # VideoMAE v2 特征维度
        self.audio_dim = 1024     # WavLM Large 特征维度
        self.audio_feature_root = "/home/pengxf/jim/work/GSR/newData/wavlm_large_features"
        
        self.lambda_bag_attn = 1.0       # 主损失权重
        self.use_instance_loss = True    # 使用实例级监督

        self.proto_m = 0.99              # 原型更新速度
        
        # 模型参数
        self.input_dim = self.visual_dim  # 向后兼容（其他模型变体使用）
        self.hidden_dim = 128            # 隐藏层维度
        self.low_dim = 64                # embedding维度
        self.num_classes = 2             # 二分类：真话/谎言
        
        # 训练参数
        self.epochs = 100
        self.batch_size = 32             # NumberGuess数据集较大，可以用更大的batch
        self.lr = 0.05
        self.weight_decay = 5e-4
        self.momentum = 0.9
        
        # 学习率调度
        self.lr_decay_epochs = [40, 70]
        self.lr_decay_rate = 0.3
        self.cosine = False
        self.warmup_epochs = 3
        
        # 其他参数
        self.print_freq = 10
        self.num_workers = 4             # NumberGuess数据量大，增加worker数
        self.seed = 42
        self.num_runs = 5                # 5折交叉验证
        self.optimizer = "sgd"
        self.top_k = 1
        
        # ====== Pairwise Ranking Loss 参数 ======
        self.rank_margin = 1.0    # MarginRankingLoss 的 margin
        self.rank_weight = 0.5    # 排序损失权重 (相对于分类损失)
        self.subject_batch_size = 4  # 被试级训练时每批被试数

        
        # ====== 损失权重参数 ======
        
        self.ortho_weight = 0.1    # 特征正交损失权重
        self.aggr_method = 'hierarchical_max'   # 聚合方法: 'hierarchical_max', 'hierarchical_mean', 'max', mean'
        # 保存路径
        self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/exper_model/newdata_audio"
        os.makedirs(self.exp_dir, exist_ok=True)

        # 消融实验配置（默认为False，即不消融）
        self.ablate_hierarchical = False      # 消融层次聚合
        self.ablate_attention = False         # 消融跨模态注意力
        self.ablate_audio_content = False     # 消融音频纯内容
        self.ablate_prototype_update = False  # 消融原型更新
        self.ablate_gate = False              # 消融门控机制
        self.ablate_rezero = False            # 消融ReZero参数
        self.ablate_audio_drop = False        # 消融Audio Drop
        self.ablate_temperature = False       # 消融注意力温度
        self.ablate_layernorm = False         # 消融LayerNorm
        self.ablate_fusion_dropout = False    # 消融融合Dropout
        self.ablate_gradient_stop = False     # 消融梯度阻断
        self.ablate_projection = False        # 消融特征投影
        self.ablate_fusion_method = 'add'     # 融合方式: 'add'(相加), 'concat'(拼接), 'multiply'(乘法)