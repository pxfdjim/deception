
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets.audio_utils import load_audio_feature


class RealLifeLOOCVDataset(Dataset):
    """
    Real-life Deception Detection 2016 数据集加载器 (留一法交叉验证版)。
    支持模态选择: 'visual' / 'audio' / 'both'
    - 视觉特征: 从pickle文件加载 (VideoMAE v2, 768维)
    - 音频特征: wavlm_large_features/trial_{lie|truth}_{NNN}.npy (WavLM Large, 1024维)
    """
    def __init__(self, samples_data, original_indices, 
                 audio_feature_root=None, modality='visual', audio_dim=1024):
        """
        Args:
            samples_data: 样本数据列表
            original_indices: 原始索引列表
            audio_feature_root: 音频特征根目录 (可选)
            modality: 模态选择 'visual'/'audio'/'both'
            audio_dim: 音频特征维度
        """
        self.current_split_data = samples_data
        self.original_indices = original_indices
        self.modality = modality
        self.audio_dim = audio_dim
        self.audio_feature_root = Path(audio_feature_root) if audio_feature_root else None

    @staticmethod
    def _load_data_from_pickle(pkl_path):
        """加载原始的 pickle 文件。"""
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ 加载特征文件失败: {pkl_path}, 错误: {e}")
            return {}

    @staticmethod
    def _flatten_data(nested_data):
        """将嵌套的字典结构扁平化为一个包含所有视频样本的列表。"""
        flat_list = []
        for person_id, person_trials in nested_data.items():
            for trial_key, trial_data in person_trials.items():
                if 'truth' in trial_key:
                    label = 0
                elif 'lie' in trial_key:
                    label = 1
                else:
                    continue  # 跳过不规范的键

                features = trial_data.get('visual')
                if features is not None:
                    flat_list.append({
                        'features': features, 
                        'label': label,
                        'id': f"{person_id}_{trial_key}"
                    })
        return flat_list

    @staticmethod
    def create_train_test_datasets(feature_path, fold_index,
                                   audio_feature_root=None, modality='visual', audio_dim=1024):
        """
        一次性创建训练集和测试集
        
        Args:
            feature_path: pickle文件路径
            fold_index: 当前fold索引 (0 到 N-1)
            audio_feature_root: 音频特征根目录 (可选)
            modality: 模态选择 'visual'/'audio'/'both'
            audio_dim: 音频特征维度
        
        Returns:
            train_dataset, test_dataset
        """
        # 加载并扁平化数据
        all_data = RealLifeLOOCVDataset._load_data_from_pickle(feature_path)
        all_samples = RealLifeLOOCVDataset._flatten_data(all_data)
        total_samples = len(all_samples)

        if not (0 <= fold_index < total_samples):
            raise ValueError(f"fold_index 必须在 0 到 {total_samples - 1} 之间，但收到了 {fold_index}")

        # 划分训练集和测试集
        test_samples = [all_samples[fold_index]]
        test_indices = [fold_index]
        
        train_samples = all_samples[:fold_index] + all_samples[fold_index+1:]
        train_indices = list(range(fold_index)) + list(range(fold_index + 1, total_samples))

        # 创建数据集实例
        train_dataset = RealLifeLOOCVDataset(train_samples, train_indices,
                                              audio_feature_root=audio_feature_root,
                                              modality=modality, audio_dim=audio_dim)
        test_dataset = RealLifeLOOCVDataset(test_samples, test_indices,
                                             audio_feature_root=audio_feature_root,
                                             modality=modality, audio_dim=audio_dim)

        # 打印统计信息
        train_labels = [s['label'] for s in train_samples]
        test_labels = [s['label'] for s in test_samples]
        print(f" Fold {fold_index + 1}/{total_samples} (模态: {modality})")
        print(f"   Train: {len(train_samples)} (Truth: {train_labels.count(0)}, Lie: {train_labels.count(1)})")
        print(f"   Test: {len(test_samples)} (Truth: {test_labels.count(0)}, Lie: {test_labels.count(1)})")

        return train_dataset, test_dataset

    def __len__(self):
        return len(self.current_split_data)

    def __getitem__(self, idx):
        video_info = self.current_split_data[idx]
        original_idx = self.original_indices[idx]
        sample_id = video_info['id']  # 形如 "person_id_trial_key"
        
        # ====== 加载视觉特征 ======
        visual_features = None
        if self.modality in ('visual', 'both'):
            features = video_info['features']
            if not isinstance(features, torch.Tensor):
                features = torch.from_numpy(features).float()
            if features.ndim == 2 and features.shape[1] == 768:
                visual_features = features
            else:
                print(f"⚠️ 视觉特征维度异常，期望 (n, 768), 实际为 {features.shape}。")
                visual_features = torch.zeros(1, 768)
        
        # ====== 加载音频特征 ======
        audio_features = None
        if self.modality in ('audio', 'both') and self.audio_feature_root:
            audio_path = self.audio_feature_root / f"{sample_id}.npy"
            if audio_path.exists():
                audio_features = load_audio_feature(audio_path, expected_dim=self.audio_dim)
            else:
                candidates = list(self.audio_feature_root.glob(f"*{sample_id}*.npy"))
                if candidates:
                    audio_features = load_audio_feature(candidates[0], expected_dim=self.audio_dim)
            if audio_features is None:
                audio_features = torch.zeros(1, self.audio_dim)
        
        return {
            'visual_features': visual_features,
            'audio_features': audio_features,
            'label': torch.tensor(video_info['label'], dtype=torch.long),
            'index': original_idx,
            'id': sample_id
        }


# ==================================================================
# ========================= 如何使用 ================================
# ==================================================================

def main():
    # =========================
    # 参数配置
    # =========================
    class Args:
        def __init__(self):
            self.feature_path = "/home/pengxf/emotion/dataset/real_life/Real-life_Deception_Detection_2016/cache/videomaev2_avg128.pkl"
            self.batch_size = 4
            self.num_workers = 0
    
    args = Args()
    
    # 先获取总样本数
    try:
        all_data = RealLifeLOOCVDataset._load_data_from_pickle(args.feature_path)
        all_samples = RealLifeLOOCVDataset._flatten_data(all_data)
        num_total_samples = len(all_samples)
        print(f"\n检测到总样本数为: {num_total_samples}。将开始留一法交叉验证...")
    except Exception as e:
        print(f"无法预加载数据集以获取样本总数，将使用默认值121。错误: {e}")
        num_total_samples = 121

    # 留一法交叉验证主循环
    for fold_idx in range(num_total_samples):
        print(f"\n========================================")
        print(f"  开始第 {fold_idx + 1} / {num_total_samples} 折交叉验证")
        print(f"========================================")

        # 一次性创建训练集和测试集
        train_dataset, test_dataset = RealLifeLOOCVDataset.create_train_test_datasets(
            args.feature_path, fold_idx
        )

        print(f"\n  训练集大小: {len(train_dataset)}")
        print(f"  测试集大小: {len(test_dataset)}")
        
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 打印第一个batch的信息
        for batch in train_loader:
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        print(f"{k}: {v.shape}")
                    else:
                        print(f"{k}: type={type(v)}")
            break
    


if __name__ == "__main__":
    main()



def real_collate_fn(batch):
    """
    Real-Life 数据集的 collate 函数 — 返回 4-tuple
    
    Returns:
        visual_list: 视觉特征列表 (元素可能为 None)
        audio_list: 音频特征列表 (元素可能为 None)
        labels: 标签张量
        ids: ID列表
    """
    visual_list = [item['visual_features'] for item in batch]
    audio_list = [item['audio_features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return visual_list, audio_list, labels, ids
