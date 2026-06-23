import os

class Args:
    """参数配置类"""
    def __init__(self):
        # 数据相关
        self.feature_path = "/home/pengxf/emotion/dataset/real_life/Real-life_Deception_Detection_2016/cache/videomaev2_avg128.pkl"
        # self.label_root = "/home/pengxf/emotion/dataset/mdpe_full_face/labels"
        # 确保在args中添加新参数:
        
        # ====== 模态设置 ======
        # 可选: 'visual' (纯视觉), 'audio' (纯音频), 'both' (视觉+音频 Late Fusion)
        self.modality = 'both'
        self.visual_dim = 768     # VideoMAE v2 特征维度
        self.audio_dim = 1024     # WavLM Large 特征维度
        self.audio_feature_root = "/home/emotion/dataset/real_life/Real-life_Deception_Detection_2016/feature/wavlm_large_features"
        
        self.lambda_main = 1
        self.proto_m = 0.999 #原型更新的动量系数
        # 模型参数
        self.input_dim = self.visual_dim  # 向后兼容（其他模型变体使用）
        self.hidden_dim = 128     # 隐藏层维度
        self.low_dim = 32        # 对比学习特征维度
        self.num_classes = 2      # 二分类：真话/谎言
        self.dropout = 0.25
        # INS参数

        self.proto_m = 0.9        # 原型更新动量
        self.partial_rate = 0.1   # 部分标签比例

        # 训练参数
        self.epochs =1
        self.batch_size = 8       # 小批次适应数据集大小
        self.lr = 0.001           # 学习率
        self.weight_decay = 5e-6
        self.momentum = 0.9
        
        # 学习率调度
        self.lr_decay_epochs = [30, 40]
        self.lr_decay_rate = 0.3
        self.cosine = False
    
        # 其他
        self.print_freq = 10
        self.num_workers = 0
        self.seed = 42
        self.num_runs = 1
        self.optimizer = "sgd"
        self.warmup_epochs = 5
        self.align_weight = 0.05    # 模态对齐损失权重 (pplg_muti 使用)
        # 保存路径
        self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/exper_model/real_new"
        os.makedirs(self.exp_dir, exist_ok=True)