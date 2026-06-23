# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import random
import numpy as np


class SEUMLDFineGrainedDataset(Dataset):
    """
    SEUMLD 谎言检测数据集加载器 (细粒度版本)。
    - 每个视频作为一个独立样本
    - 标签按视频粒度：name,label (如 005_01,0)
    - 0=真实(truth), 1=谎言(lie)
    - 支持视觉+音频多模态
    """
    def __init__(self, feature_root, label_path, fold_path, fold_index=0, split='train',
                 audio_feature_root=None, modality='visual', audio_dim=1024):
        self.feature_root = Path(feature_root)
        self.audio_feature_root = Path(audio_feature_root) if audio_feature_root else None
        self.modality = modality
        self.audio_dim = audio_dim
        self.split = split
        self.fold_index = fold_index
        print(f"[*] 正在加载 {split} 数据集 (第 {fold_index + 1} 折)，细粒度模式，模态: {modality}...")
        
        self.folds = self._load_folds(fold_path)
        
        # 加载细粒度标签：video_name -> label
        self.video_labels_map = self._load_video_labels(label_path)
        
        # 获取当前split的person_ids
        self.person_ids = self._get_person_ids_for_split()
        
        # 构建样本列表：[(feature_path, audio_feature_path, label, video_name), ...]
        self.samples = self._build_samples(self.person_ids)
        
        print(f"[*] {split} 数据集加载完成，共 {len(self.samples)} 个视频样本。")
        self._print_label_distribution()

    def _load_folds(self, fold_path):
        """加载交叉验证折"""
        try:
            df = pd.read_csv(fold_path)
            return [[str(sid).zfill(3) for sid in df[col].dropna().astype(int)] for col in df.columns]
        except Exception as e:
            print(f"❌ 加载交叉验证文件失败: {e}")
            return [[] for _ in range(5)]

    def _load_video_labels(self, label_path):
        """加载细粒度标签文件：video_name -> label"""
        try:
            df = pd.read_csv(label_path)
            # 假设列名为 'name' 和 'label'
            df.columns = [c.strip() for c in df.columns]
            # 创建字典：video_name -> label
            return pd.Series(df.label.values, index=df.name).to_dict()
        except Exception as e:
            print(f"❌ 加载标签文件失败: {e}")
            return {}

    def _get_person_ids_for_split(self):
        """根据split获取person_ids"""
        if self.split == 'test':
            return self.folds[self.fold_index]
        # train: 其他所有折
        train_person_ids = []
        for i, fold_subjects in enumerate(self.folds):
            if i != self.fold_index:
                train_person_ids.extend(fold_subjects)
        return train_person_ids

    def _build_samples(self, person_ids):
        """构建样本列表：遍历person_ids，找到所有视频特征文件
        
        注意：缺少音频特征的样本会被保留，在 __getitem__ 中会返回零向量
        """
        samples = []
        missing_labels = 0
        missing_audio = 0
        
        for person_id in person_ids:
            person_dir = self.feature_root / person_id
            if not person_dir.is_dir():
                continue
            
            # 遍历该person目录下的所有.pt文件
            for feature_path in person_dir.glob('*.pt'):
                # 从文件名提取video_name: 005_12.pt -> 005_12
                video_name = feature_path.stem
                
                # 检查标签是否存在
                if video_name not in self.video_labels_map:
                    missing_labels += 1
                    if missing_labels <= 5:
                        print(f"⚠️  视频 {video_name} 在标签文件中不存在，跳过")
                    continue
                
                label = self.video_labels_map[video_name]
                
                # 查找音频特征（缺少时在 __getitem__ 中用零向量填充）
                audio_feature_path = None
                if self.audio_feature_root is not None and self.modality in ['audio', 'both']:
                    # 音频特征路径结构: audio_feature_root/person_id/video_name.npy
                    audio_path = self.audio_feature_root / person_id / f"{video_name}.npy"
                    if audio_path.exists():
                        audio_feature_path = audio_path
                    else:
                        missing_audio += 1
                        # 静默处理，不打印警告
                
                samples.append((feature_path, audio_feature_path, label, video_name))
        
        if missing_labels > 5:
            print(f"⚠️  共有 {missing_labels} 个视频缺少标签")
        if missing_audio > 0:
            print(f"ℹ️  共有 {missing_audio} 个视频缺少音频特征（将用零向量填充）")
        
        return samples

    def _print_label_distribution(self):
        """打印标签分布"""
        if not self.samples:
            return
        labels = [label for _, _, label, _ in self.samples]
        print(f"    标签分布 -> 真实 (0): {labels.count(0)}, 谎言 (1): {labels.count(1)}")

    @staticmethod
    def create_train_test_datasets(feature_root, label_path, fold_path, fold_index,
                                   audio_feature_root=None, modality='visual', audio_dim=1024):
        """
        一次性创建训练集和测试集
        
        Returns:
            train_dataset, test_dataset
        """
        train_dataset = SEUMLDFineGrainedDataset(
            feature_root, label_path, fold_path, fold_index, 'train',
            audio_feature_root=audio_feature_root, modality=modality, audio_dim=audio_dim
        )
        test_dataset = SEUMLDFineGrainedDataset(
            feature_root, label_path, fold_path, fold_index, 'test',
            audio_feature_root=audio_feature_root, modality=modality, audio_dim=audio_dim
        )
        return train_dataset, test_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, audio_feature_path, label, video_name = self.samples[idx]
        
        # 加载视觉特征
        try:
            # 加载特征：shape应该是 (num_segments, 768)
            features = torch.load(feature_path, map_location='cpu', weights_only=True)
            
            # 确保特征格式正确
            if features.ndim != 2 or features.shape[1] != 768:
                print(f"⚠️  特征形状异常 {feature_path}: {features.shape}")
                features = torch.zeros(1, 768)
        except Exception as e:
            print(f"❌ 加载特征文件失败 {feature_path}: {e}")
            features = torch.zeros(1, 768)
        
        # 加载音频特征
        audio_features = None
        if self.modality in ['audio', 'both']:
            if audio_feature_path is not None and audio_feature_path.exists():
                try:
                    audio_features = np.load(audio_feature_path)
                    # WavLM large 输出: (seq_len, 1024) -> 取均值得到全局特征
                    if audio_features.ndim == 2:
                        audio_features = np.mean(audio_features, axis=0)  # (1024,)
                    audio_features = torch.from_numpy(audio_features).float()
                except Exception as e:
                    print(f"❌ 加载音频特征失败 {audio_feature_path}: {e}")
                    audio_features = torch.zeros(self.audio_dim)
            else:
                audio_features = torch.zeros(self.audio_dim)
        
        return {
            'features': features,
            'audio_features': audio_features,
            'label': label,
            'video_name': video_name
        }


# ==============================================================================
# 2. 工厂函数与collate_fn
# ==============================================================================
def finegrained_collate_fn(batch):
    """
    细粒度collate函数 - 仅视觉模态
    batch中每个item是一个视频的特征
    """
    features_list = [item['features'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    video_names = [item['video_name'] for item in batch]
    return features_list, None, labels, video_names


def seumld_multimodal_collate_fn(batch):
    """
    SEUMLD多模态collate函数
    batch中每个item是一个视频的视觉和音频特征
    
    Returns:
        visual_list: 视觉特征列表
        audio_list: 音频特征列表 (每个样本一个全局特征向量)
        labels: 标签tensor
        video_names: 视频名称列表
    """
    visual_list = [item['features'] for item in batch]
    audio_list = [item['audio_features'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    video_names = [item['video_name'] for item in batch]
    
    # 处理音频特征：如果为None，创建零向量
    audio_list_processed = []
    for audio_feat in audio_list:
        if audio_feat is None:
            audio_list_processed.append(torch.zeros(1024))
        else:
            audio_list_processed.append(audio_feat)
    
    return visual_list, audio_list_processed, labels, video_names


def create_finegrained_dataloaders(args, fold_idx, collate_fn):
    """
    创建细粒度的 train/test DataLoader
    """
    train_dataset = SEUMLDFineGrainedDataset(
        args.feature_root, args.label_path, args.fold_path, fold_idx, 'train',
        audio_feature_root=getattr(args, 'audio_feature_root', None),
        modality=getattr(args, 'modality', 'visual'),
        audio_dim=getattr(args, 'audio_dim', 1024)
    )
    test_dataset = SEUMLDFineGrainedDataset(
        args.feature_root, args.label_path, args.fold_path, fold_idx, 'test',
        audio_feature_root=getattr(args, 'audio_feature_root', None),
        modality=getattr(args, 'modality', 'visual'),
        audio_dim=getattr(args, 'audio_dim', 1024)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


# ==============================================================================
# 3. 主执行块 (用于测试)
# ==============================================================================
if __name__ == '__main__':
    # 使用真实数据路径进行测试
    real_feature_root = "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames"
    real_label_path = "/home/pengxf/emotion/dataset/SEUMLD/Labels/Fine-grained-labels.csv"
    real_fold_path = "/home/pengxf/emotion/dataset/SEUMLD/Original/5fold_list.csv"
    real_audio_root = "/home/pengxf/jim/work/GSR/SEU_Audio/wavlm_large_features"
    
    print("="*80)
    print("开始测试【细粒度】数据加载流程 - 5折交叉验证 (多模态)")
    print(f"视觉特征路径: {real_feature_root}")
    print(f"音频特征路径: {real_audio_root}")
    print(f"标签路径: {real_label_path}")
    print("="*80)

    try:
        # 创建一个简单的args对象
        class Args:
            feature_root = real_feature_root
            label_path = real_label_path
            fold_path = real_fold_path
            audio_feature_root = real_audio_root
            audio_dim = 1024
            modality = 'both'
            batch_size = 4
            num_workers = 0
        
        args = Args()
        
        # 测试所有5折
        print("\n" + "="*80)
        print("测试所有5折的数据加载")
        print("="*80)
        
        all_fold_stats = []
        
        for fold_idx in range(5):
            print(f"\n{'='*80}")
            print(f"第 {fold_idx + 1} 折")
            print(f"{'='*80}")
            
            train_loader, test_loader, train_dataset, test_dataset = create_finegrained_dataloaders(
                args, fold_idx=fold_idx, collate_fn=seumld_multimodal_collate_fn
            )
            
            # 统计信息
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            
            # 统计训练集标签分布
            train_labels = [label for _, _, label, _ in train_dataset.samples]
            train_truth = train_labels.count(0)
            train_lie = train_labels.count(1)
            
            # 统计测试集标签分布
            test_labels = [label for _, _, label, _ in test_dataset.samples]
            test_truth = test_labels.count(0)
            test_lie = test_labels.count(1)
            
            print(f"\n训练集:")
            print(f"  - 总样本数: {train_size}")
            print(f"  - 真实(0): {train_truth} ({train_truth/train_size*100:.1f}%)")
            print(f"  - 谎言(1): {train_lie} ({train_lie/train_size*100:.1f}%)")
            print(f"  - Batch数: {len(train_loader)}")
            
            print(f"\n测试集:")
            print(f"  - 总样本数: {test_size}")
            print(f"  - 真实(0): {test_truth} ({test_truth/test_size*100:.1f}%)")
            print(f"  - 谎言(1): {test_lie} ({test_lie/test_size*100:.1f}%)")
            print(f"  - Batch数: {len(test_loader)}")
            
            # 保存统计信息
            all_fold_stats.append({
                'fold': fold_idx + 1,
                'train_size': train_size,
                'train_truth': train_truth,
                'train_lie': train_lie,
                'test_size': test_size,
                'test_truth': test_truth,
                'test_lie': test_lie
            })
            
            # 测试加载一个batch
            if fold_idx == 0:
                print(f"\n{'='*80}")
                print("测试第1折的数据加载（示例）")
                print(f"{'='*80}")
                
                visual_list, audio_list, labels, video_names = next(iter(train_loader))
                
                print(f"\n训练集 Batch 示例:")
                print(f"  - Batch大小: {len(visual_list)}")
                
                for i, (v_feat, a_feat, vname) in enumerate(zip(visual_list[:2], audio_list[:2], video_names[:2])):
                    print(f"  - 视频 {i+1} ({vname}):")
                    print(f"      视觉特征形状: {v_feat.shape}")
                    print(f"      音频特征形状: {a_feat.shape}")
                
                print(f"  - 标签: {labels.numpy()}")
                
                # 测试test_loader
                test_visual_list, test_audio_list, test_labels, test_video_names = next(iter(test_loader))
                print(f"\n测试集 Batch 示例:")
                print(f"  - Batch大小: {len(test_visual_list)}")
                print(f"  - 视频名称: {test_video_names[:2]}")
                print(f"  - 标签: {test_labels.numpy()}")
        
        # 打印总体统计
        print(f"\n{'='*80}")
        print("5折交叉验证总体统计")
        print(f"{'='*80}")
        
        print(f"\n{'折':<6} {'训练集':<10} {'真实':<8} {'谎言':<8} {'测试集':<10} {'真实':<8} {'谎言':<8}")
        print("-" * 80)
        
        for stats in all_fold_stats:
            print(f"{stats['fold']:<6} "
                  f"{stats['train_size']:<10} "
                  f"{stats['train_truth']:<8} "
                  f"{stats['train_lie']:<8} "
                  f"{stats['test_size']:<10} "
                  f"{stats['test_truth']:<8} "
                  f"{stats['test_lie']:<8}")
        
        # 计算平均值
        avg_train = sum(s['train_size'] for s in all_fold_stats) / 5
        avg_test = sum(s['test_size'] for s in all_fold_stats) / 5
        total_samples = all_fold_stats[0]['train_size'] + all_fold_stats[0]['test_size']
        
        print("-" * 80)
        print(f"平均   {avg_train:<10.1f} {'':<16} {avg_test:<10.1f}")
        print(f"\n总样本数: {total_samples}")
        print(f"平均训练集大小: {avg_train:.1f}")
        print(f"平均测试集大小: {avg_test:.1f}")
        print(f"训练/测试比例: {avg_train/avg_test:.2f}")
        
        print("\n" + "="*80)
        print("[SUCCESS] 所有5折数据加载测试完成！")
        print("="*80)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] 在数据加载过程中发生错误！")
        traceback.print_exc()
