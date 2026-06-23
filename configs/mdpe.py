import os

class Args:
    """参数配置类"""
    def __init__(self):
        # 数据相关
        self.feature_root = "/home/pengxf/work/TDD/Video_MAEV2/data"
        self.label_root = "/home/pengxf/emotion/dataset/mdpe_full_face/labels"
        # 确保在args中添加新参数:
        self.lambda_main = 1

        # 模型参数
        self.input_dim = 768      # VideoMAE特征维度
        self.hidden_dim = 128     # 隐藏层维度
        self.low_dim = 64      # 对比学习特征维度
        self.num_classes = 2      # 二分类：真话/谎言
        


        self.proto_m = 0.9        # 原型更新动量
        self.partial_rate = 0.1   # 部分标签比例

        
        # 训练参数
        self.epochs = 50
        self.batch_size = 8       # 小批次适应数据集大小
        self.lr = 0.0005           # 学习率
        self.weight_decay = 1e-3
        self.momentum = 0.9
        
        # 学习率调度
        self.lr_decay_epochs = [30, 40]
        self.lr_decay_rate = 0.3
        self.cosine = True
        
        # 其他
        self.print_freq = 10
        self.num_workers = 1
        self.seed = 42
        self.num_runs = 1
        self.optimizer = "sgd"
        self.warmup_epochs = 5
        # 保存路径
        self.exp_dir = "/home/pengxf/work/TDD/Video_MAEV2/Deception/exper_model/new_mdpe"
        os.makedirs(self.exp_dir, exist_ok=True)