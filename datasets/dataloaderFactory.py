# file: dataloader_factory.py

from torch.utils.data import DataLoader

from .mdpe import MdpeDetectionDataset 
from .real_life import RealLifeLOOCVDataset
from .dolos import DOLOSDataset, dolos_collate_fn
from .seumld import SEUMLDFineGrainedDataset, finegrained_collate_fn
from .newData import NumberGuessDataset, question_collate_fn as newdata_collate_fn


def create_mdpe_dataloaders(feature_root, collate_fn,label_root, batch_size=16, partial_rate=0.1, 
                          num_workers=4 ):
    """
    创建INS训练所需的数据加载器
    
    Args:
        feature_root: 特征根目录
        label_root: 标签根目录
        batch_size: 批次大小
        partial_rate: 部分标签比例
        num_workers: 数据加载进程数
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    
    print(" Creating INS DataLoaders...")
    
    # 一次性创建训练集和测试集
    train_dataset, test_dataset = MdpeDetectionDataset.create_train_test_datasets(
        feature_root, label_root, partial_rate=partial_rate
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # INS需要固定批次大小
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f" DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def create_loocv_dataloaders(feature_path, collate_fn, fold_index, batch_size, num_workers,
                             audio_feature_root=None, modality='visual', audio_dim=1024):
    """
    为 Real-Life Deception LOOCV 训练创建数据加载器。

    Args:
        feature_path (str): 指向 .pkl 视觉特征文件的路径。
        collate_fn (callable): 用于整理批数据的函数。
        fold_index (int): 当前使用的折数 (0 to N-1)。
        batch_size (int): 训练集的批次大小。
        num_workers (int): 数据加载进程数。
        audio_feature_root (str): 音频特征根目录 (可选)。
        modality (str): 模态选择 'visual'/'audio'/'both'。
        audio_dim (int): 音频特征维度。

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    print(f"===== [Factory] Creating DataLoaders for Fold {fold_index + 1} (模态: {modality}) =====")
    
    # 一次性创建训练集和测试集
    train_dataset, test_dataset = RealLifeLOOCVDataset.create_train_test_datasets(
        feature_path, fold_index,
        audio_feature_root=audio_feature_root,
        modality=modality, audio_dim=audio_dim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # 确保批次大小一致
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 对于 LOOCV，测试集的 batch_size 必须是 1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"[*] [Factory] DataLoaders created successfully for Fold {fold_index + 1}.")
    
    return train_loader, test_loader, train_dataset, test_dataset


def create_dolos_dataloaders(feature_root, fold_path, collate_fn, fold_index=0, 
                             batch_size=16, num_workers=4,
                             audio_feature_root=None, modality='visual', audio_dim=1024):
    """
    创建 DOLOS 数据集的数据加载器（3折交叉验证）
    
    Args:
        feature_root: 视觉特征根目录
        fold_path: fold文件目录路径
        collate_fn: 用于整理批数据的函数
        fold_index: 当前fold索引 0-2
        batch_size: 批次大小
        num_workers: 数据加载进程数
        audio_feature_root: 音频特征根目录 (可选)
        modality: 模态选择 'visual'/'audio'/'both'
        audio_dim: 音频特征维度
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    print(f"===== [Factory] Creating DOLOS DataLoaders for Fold {fold_index + 1} (模态: {modality}) =====")
    
    # 一次性创建训练集和测试集
    train_dataset, test_dataset = DOLOSDataset.create_train_test_datasets(
        feature_root, fold_path, fold_index,
        audio_feature_root=audio_feature_root,
        modality=modality, audio_dim=audio_dim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"[*] [Factory] DOLOS DataLoaders created successfully for Fold {fold_index + 1}.")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def create_seumld_dataloaders(feature_root, label_path, fold_path, collate_fn, 
                              fold_index=0, batch_size=16, num_workers=4,
                              audio_feature_root=None, modality='visual', audio_dim=1024):
    """
    创建 SEUMLD 细粒度数据集的数据加载器（5折交叉验证）
    
    Args:
        feature_root: 视觉特征根目录
        label_path: 标签文件路径
        fold_path: 交叉验证折文件路径
        collate_fn: 用于整理批数据的函数
        fold_index: 当前折索引 (0-4)
        batch_size: 批次大小
        num_workers: 数据加载进程数
        audio_feature_root: 音频特征根目录 (可选)
        modality: 模态选择 'visual'/'audio'/'both'
        audio_dim: 音频特征维度
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    print(f"===== [Factory] Creating SEUMLD DataLoaders for Fold {fold_index + 1} (模态: {modality}) =====")
    
    # 一次性创建训练集和测试集
    train_dataset, test_dataset = SEUMLDFineGrainedDataset.create_train_test_datasets(
        feature_root, label_path, fold_path, fold_index,
        audio_feature_root=audio_feature_root, modality=modality, audio_dim=audio_dim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"[*] [Factory] SEUMLD DataLoaders created successfully for Fold {fold_index + 1}.")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset, test_dataset



def create_newdata_dataloaders(feature_dir, fold_path, collate_fn, fold_index=0, 
                               batch_size=32, num_workers=4, include_baseline=False, 
                               target_only=True, shuffle_train=True,
                               audio_feature_dir=None, modality='visual', audio_dim=1024):
    """
    创建 NumberGuess 数据集的数据加载器（5折交叉验证）
    
    Args:
        feature_dir: 视觉特征根目录
        fold_path: 交叉验证折文件路径
        collate_fn: 用于整理批数据的函数
        fold_index: 当前折索引 (0-4)
        batch_size: 批次大小
        num_workers: 数据加载进程数
        include_baseline: 是否包含baseline问题
        target_only: 是否只使用target问题
        shuffle_train: 是否打乱训练集
        audio_feature_dir: 音频特征根目录 (可选)
        modality: 模态选择 'visual'/'audio'/'both'
        audio_dim: 音频特征维度
    
    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    from pathlib import Path
    import csv
    
    print(f"===== [Factory] Creating NumberGuess DataLoaders for Fold {fold_index + 1} (模态: {modality}) =====")
    
    # 加载fold划分
    fold_map = {}
    with open(fold_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_map[row["id"].strip()] = int(row["fold"])
    
    # 划分训练集和测试集被试
    train_subjs = [s for s, f in fold_map.items() if f != fold_index + 1]  # fold从1开始
    test_subjs = [s for s, f in fold_map.items() if f == fold_index + 1]
    
    # 创建数据集
    train_dataset = NumberGuessDataset(
        train_subjs, 
        feature_dir=Path(feature_dir),
        include_baseline=include_baseline,
        target_only=target_only,
        fold_map=fold_map,
        audio_feature_dir=audio_feature_dir,
        modality=modality,
        audio_dim=audio_dim
    )
    
    test_dataset = NumberGuessDataset(
        test_subjs,
        feature_dir=Path(feature_dir),
        include_baseline=include_baseline,
        target_only=target_only,
        fold_map=fold_map,
        audio_feature_dir=audio_feature_dir,
        modality=modality,
        audio_dim=audio_dim
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"[*] [Factory] NumberGuess DataLoaders created successfully for Fold {fold_index + 1}.")
    print(f"   Train batches: {len(train_loader)}, samples: {len(train_dataset)}")
    print(f"   Test batches: {len(test_loader)}, samples: {len(test_dataset)}")
    print(f"   Train label distribution: {train_dataset.get_label_distribution()}")
    print(f"   Test label distribution: {test_dataset.get_label_distribution()}")
    
    return train_loader, test_loader, train_dataset, test_dataset
