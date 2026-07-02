import os

class Args:
    """SEUMLD 数据集参数配置类 - 5折交叉验证"""
    def __init__(self):
        # SEUMLD数据相关
        self.feature_root = "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames"
        self.label_path = "/home/pengxf/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv"
        self.fold_path = "/home/pengxf/emotion/dataset/SEUMLD/Original/5fold_list.csv"
        
        # 音频特征路径
        self.audio_feature_root = "/home/pengxf/jim/work/GSR/SEU_Audio/wavlm_large_features"
        self.audio_dim = 1024  # wavlm_large 特征维度
        
        # 模态选择: 'visual', 'audio', 'both'
        self.modality = 'both'
        
        self.lambda_bag_attn = 1.0      # 主损失权重

        self.proto_m = 0.95             # 适中的原型更新速度
        
        # 正交损失权重
        self.ortho_weight = 0.1
        
        # 模型参数 - 适配层次化聚合架构
        self.input_dim = 768            # VideoMAE特征维度
        self.hidden_dim = 256           # 适中的隐藏层（不要太大也不要太小）
        self.low_dim = 64               # 适中的embedding维度
        self.num_classes = 2            # 二分类：真话/谎言
        self.use_instance_loss = True
        self.instance_loss_weight = 0.1
        self.positive_instance_topk_ratio = 0.25
        # Default clean SEU setting: conservative top-k prototype update + cluster top-k pooling.
        self.use_topk_proto_update = True
        self.use_conservative_topk_proto_update = True
        self.topk_proto_ratio = 0.25
        self.topk_proto_threshold = 0.6
        self.topk_proto_warmup_epochs = 5
        self.use_cluster_topk_mean_pooling = True
        self.cluster_topk_mean_ratio = 0.5
        
        
        # 训练参数
        self.epochs = 100
        self.batch_size = 8
        self.lr = 0.05                 # 增加学习率（从0.01到0.05）
        self.weight_decay = 5e-4        # 适中的正则化
        self.momentum = 0.9
        
        # 学习率调度
        self.lr_decay_epochs = [40, 70]  # 更晚开始衰减
        self.lr_decay_rate = 0.3
        self.cosine = False
        self.warmup_epochs = 3           # 添加warmup帮助初期学习
        
        # 其他参数
        self.aggr_method = 'hierarchical_mean'  # 聚合方法: 'hierarchical_max', 'hierarchical_mean', 'max', mean'
        self.print_freq = 10
        self.num_workers = 1
        self.seed = 42
        self.num_runs = 5
        self.optimizer = "sgd"           # 明确设置为SGD
        self.eval_every = 1
        self.best_epoch_objective = "acc_f1"
        self.best_epoch_acc_tolerance = 0.0
        self.early_stop_patience = 0
        self.early_stop_min_epochs = 0
        self.disable_tqdm = False
  
        
        # 保存路径
        self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/seu_ortho"
        os.makedirs(self.exp_dir, exist_ok=True)
