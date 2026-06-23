# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets.audio_utils import load_audio_feature


class DOLOSDataset(Dataset):
    """
    DOLOS 谎言检测数据集加载器（3折交叉验证）
    - 视觉特征: features/videomaev2/{video_name}.pt  (VideoMAE v2, 768维)
    - 音频特征: wavlm_large_features/{video_name}.npy (WavLM Large, 1024维)
    - 标签从文件名提取：_lie, _truth, _true, deception
    - 0=truth(真实), 1=lie(谎言)
    - 使用3折交叉验证划分
    - 支持模态选择: 'visual' / 'audio' / 'both'
    """
    def __init__(self, feature_root, fold_path, fold_index, split='train',
                 audio_feature_root=None, modality='visual', audio_dim=1024):
        self.feature_root = Path(feature_root)
        self.fold_path = Path(fold_path)
        self.fold_index = fold_index
        self.split = split
        self.modality = modality
        self.audio_dim = audio_dim
        self.audio_feature_root = Path(audio_feature_root) if audio_feature_root else None
        
        print(f"[*] 正在加载 DOLOS {split} 数据集 (第 {fold_index + 1} 折, 模态: {modality})...")
        self.samples = self._load_from_fold_file()
        
        print(f"[*] DOLOS {split} 数据集加载完成，共 {len(self.samples)} 个视频样本。")
        self._print_label_distribution()

    def _load_from_fold_file(self):
        """从fold文件加载样本列表"""
        # 构建文件名：train_fold1.csv 或 test_fold1.csv
        fold_file = self.fold_path / f"{self.split}_fold{self.fold_index + 1}.csv"
        
        if not fold_file.exists():
            print(f"❌ Fold文件不存在: {fold_file}")
            return []
        
        # 读取CSV文件：格式为 video_name,label,gender
        samples = []
        missing_count = 0
        
        with open(fold_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 解析CSV行
                parts = line.split(',')
                if len(parts) < 2:
                    print(f"⚠️  第{line_num}行格式错误: {line}")
                    continue
                
                video_name = parts[0].strip()
                label_str = parts[1].strip().lower()
                
                # 从CSV的label列获取标签
                # 支持多种格式：deception, lie, truth, true
                if label_str in ['deception', 'lie','Deception','Lie']:
                    label = 1  # 谎言
                elif label_str in ['truth', 'true', 'Truth', 'True']:
                    label = 0  # 真实
                else:
                    print(f"⚠️  第{line_num}行标签未知: {label_str}")
                    continue
                
                # 构建特征文件路径
                feature_path = self.feature_root / f"{video_name}.pt"
                
                if not feature_path.exists():
                    missing_count += 1
                    if missing_count <= 5:
                        print(f"⚠️  特征文件不存在: {feature_path.name}")
                    continue
                
                samples.append((feature_path, label, video_name))
        
        print(f"从 {fold_file.name} 读取了 {len(samples)} 个样本")
        
        if missing_count > 5:
            print(f"⚠️  共有 {missing_count} 个视频缺少特征文件")
        
        return samples

    @staticmethod
    def create_train_test_datasets(feature_root, fold_path, fold_index,
                                   audio_feature_root=None, modality='visual', audio_dim=1024):
        """
        一次性创建训练集和测试集（3折交叉验证）
        
        Args:
            feature_root: 视觉特征根目录
            fold_path: fold文件目录路径
            fold_index: 当前fold索引 0-2
            audio_feature_root: 音频特征根目录 (可选)
            modality: 模态选择 'visual'/'audio'/'both'
            audio_dim: 音频特征维度
        
        Returns:
            train_dataset, test_dataset
        """
        train_dataset = DOLOSDataset(feature_root, fold_path, fold_index, split='train',
                                     audio_feature_root=audio_feature_root,
                                     modality=modality, audio_dim=audio_dim)
        test_dataset = DOLOSDataset(feature_root, fold_path, fold_index, split='test',
                                    audio_feature_root=audio_feature_root,
                                    modality=modality, audio_dim=audio_dim)
        
        return train_dataset, test_dataset

    def _print_label_distribution(self):
        """打印标签分布"""
        if not self.samples:
            return
        labels = [label for _, label, _ in self.samples]
        truth_count = labels.count(0)
        lie_count = labels.count(1)
        total = len(labels)
        print(f"    标签分布 -> 真实 (0): {truth_count} ({truth_count/total*100:.1f}%), "
              f"谎言 (1): {lie_count} ({lie_count/total*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label, video_name = self.samples[idx]
        
        # ====== 加载视觉特征 ======
        visual_features = None
        if self.modality in ('visual', 'both'):
            try:
                visual_features = torch.load(feature_path, map_location='cpu', weights_only=True)
                if visual_features.ndim != 2 or visual_features.shape[1] != 768:
                    print(f"⚠️  视觉特征形状异常 {feature_path}: {visual_features.shape}")
                    visual_features = torch.zeros(1, 768)
            except Exception as e:
                print(f"❌ 加载视觉特征失败 {feature_path}: {e}")
                visual_features = torch.zeros(1, 768)
        
        # ====== 加载音频特征 ======
        audio_features = None
        if self.modality in ('audio', 'both') and self.audio_feature_root:
            audio_path = self.audio_feature_root / f"{video_name}.npy"
            audio_features = load_audio_feature(audio_path, expected_dim=self.audio_dim)
            if audio_features is None:
                audio_features = torch.zeros(1, self.audio_dim)
        
        return {
            'visual_features': visual_features,
            'audio_features': audio_features,
            'label': label,
            'video_name': video_name
        }


# ==============================================================================
# collate_fn
# ==============================================================================
def dolos_collate_fn(batch):
    """
    DOLOS collate函数 — 返回 4-tuple: (visual_list, audio_list, labels, names)
    visual_list/audio_list 中的元素可能为 None（当该模态未启用时）
    """
    visual_list = [item['visual_features'] for item in batch]
    audio_list = [item['audio_features'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    video_names = [item['video_name'] for item in batch]
    return visual_list, audio_list, labels, video_names


# ==============================================================================
# 主执行块 (用于测试)
# ==============================================================================
if __name__ == '__main__':
    # 使用真实数据路径进行测试
    real_feature_root = "/home/pengxf/emotion/dataset/DOLOS/features/videomaev2"
    real_fold_path = "/home/pengxf/emotion/dataset/DOLOS/Training_Protocols"
    
    print("="*80)
    print("开始测试 DOLOS 数据加载流程（3折交叉验证）")
    print(f"特征路径: {real_feature_root}")
    print(f"Fold路径: {real_fold_path}")
    print("="*80)

    try:
        # 测试所有3折
        for fold_idx in range(3):
            print(f"\n{'='*80}")
            print(f"第 {fold_idx + 1} 折")
            print(f"{'='*80}")
            
            # 创建训练集和测试集
            train_dataset, test_dataset = DOLOSDataset.create_train_test_datasets(
                real_feature_root, real_fold_path, fold_idx
            )
            
            # 创建DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=dolos_collate_fn
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=dolos_collate_fn
            )
            
            print(f"\n训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个batches")
            print(f"测试集: {len(test_dataset)} 个样本, {len(test_loader)} 个batches")
            
            # 只在第1折测试加载一个batch
            if fold_idx == 0:
                print(f"\n测试第1折的数据加载...")
                features_list, labels, video_names = next(iter(train_loader))
                
                print(f"\nBatch 信息:")
                print(f"  - Batch大小: {len(features_list)}")
                print(f"  - 标签: {labels.numpy()}")
                print(f"  - 视频名称: {video_names[:2]}")  # 只显示前2个
                
                for i, features in enumerate(features_list[:2]):  # 只显示前2个
                    print(f"  - 视频 {i+1} 特征形状: {features.shape}")
        
        print("\n" + "="*80)
        print("[SUCCESS] DOLOS 3折交叉验证数据加载测试完成！")
        print("="*80)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] 在数据加载过程中发生错误！")
        traceback.print_exc()
