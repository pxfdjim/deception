import numpy as np
import torch
import torch.utils.data as data_utils
import os
import pickle
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd



class MdpeDetectionDataset(Dataset):
    """
    MDPE谎言检测数据集（支持部分标签噪声）
    """

    def __init__(self, video_list, partial_rate=0.0):
        """
        Args:
            video_list: 视频数据列表
            partial_rate: 部分标签噪声比例
        """
        self.video_list = video_list
        self.partial_rate = partial_rate
        
        # 如果需要添加噪声
        if partial_rate > 0:
            self._add_partial_noise()
        
        print(f" Dataset initialized with {len(self.video_list)} videos (partial_rate={partial_rate})")
    
    def _add_partial_noise(self):
        """为部分样本添加标签噪声"""
        num_samples = len(self.video_list)
        num_noisy = int(num_samples * self.partial_rate)
        
        if num_noisy > 0:
            noisy_indices = random.sample(range(num_samples), num_noisy)
            for idx in noisy_indices:
                # 翻转标签
                self.video_list[idx]['label'] = 1 - self.video_list[idx]['label']
            
            print(f" Added noise to {num_noisy}/{num_samples} samples")
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_info = self.video_list[idx]
        
        try:
            features = torch.load(video_info['feature_path'], map_location='cpu', weights_only=False)
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
        except Exception as e:
            print(f"❌ Error loading {video_info['feature_path']}: {e}")
            features = torch.zeros(8, 768)
        
        return {
            'features': features,
            'true_labels': torch.tensor(video_info['label'], dtype=torch.long),
            'person_id': video_info['person_id'],
            'question_order': video_info['question_order'],
            'question_id': video_info['question_id'],
            'original_label': video_info['original_label'],
            'file_path': str(video_info['feature_path']),
            'index': idx
        }


    @staticmethod
    def _load_all_video_data(feature_root, label_root):
        """加载所有视频数据"""
        video_data = []
        person_labels = {}
        
        # 首先加载所有标签
        for label_file in label_root.glob('*.csv'):
            person_id = label_file.stem
            
            try:
                df = pd.read_csv(label_file)
                person_labels[person_id] = {}
                
                for _, row in df.iterrows():
                    question_order = int(row.iloc[0])
                    question_id = int(row.iloc[1])
                    label_value = int(row.iloc[-1])
                    binary_label = 1 if label_value != 0 else 0
                    
                    person_labels[person_id][question_order] = {
                        'question_id': question_id,
                        'label': binary_label,
                        'original_label': label_value
                    }
            except Exception as e:
                print(f" Error loading labels for person {person_id}: {e}")
        
        # 然后匹配特征文件
        for person_dir in sorted(feature_root.iterdir()):
            if person_dir.is_dir() and person_dir.name.isdigit():
                person_id = person_dir.name
                
                if person_id not in person_labels:
                    continue
                
                for feature_file in person_dir.glob('*.pt'):
                    try:
                        parts = feature_file.stem.split('-')
                        if len(parts) >= 3:
                            parsed_person = parts[0]
                            question_order = int(parts[1])
                            question_id = int(parts[2])
                            
                            if (parsed_person == person_id and 
                                question_order in person_labels[person_id] and
                                person_labels[person_id][question_order]['question_id'] == question_id):
                                
                                video_data.append({
                                    'feature_path': feature_file,
                                    'person_id': person_id,
                                    'question_order': question_order,
                                    'question_id': question_id,
                                    'label': person_labels[person_id][question_order]['label'],
                                    'original_label': person_labels[person_id][question_order]['original_label']
                                })
                    except Exception as e:
                        print(f"❌ Error processing {feature_file}: {e}")
        
        print(f" Loaded {len(video_data)} videos")
        return video_data
    
    @staticmethod
    def _group_by_person(video_data):
        """按人分组视频数据"""
        person_videos = {}
        
        for video in video_data:
            person_id = video['person_id']
            if person_id not in person_videos:
                person_videos[person_id] = {'label_0': [], 'label_1': []}
            
            if video['label'] == 0:
                person_videos[person_id]['label_0'].append(video)
            else:
                person_videos[person_id]['label_1'].append(video)
        
        valid_persons = []
        for person_id, videos in person_videos.items():
            label_0_count = len(videos['label_0'])
            label_1_count = len(videos['label_1'])
            
            if label_0_count >= 3 and label_1_count >= 2:
                valid_persons.append(person_id)
        
        print(f"📊 Valid persons: {len(valid_persons)}/{len(person_videos)}")
        
        filtered_person_videos = {pid: person_videos[pid] for pid in valid_persons}
        return filtered_person_videos
    
    @staticmethod
    def create_train_test_datasets(feature_root, label_root, partial_rate=0.1):
        """
        一次性创建训练集和测试集
        
        Args:
            feature_root: 特征根目录
            label_root: 标签根目录
            partial_rate: 训练集的部分标签比例（测试集不添加噪声）
        
        Returns:
            train_dataset, test_dataset
        """
        print(" Creating train and test datasets...")
        
        feature_root = Path(feature_root)
        label_root = Path(label_root)
        
        # 加载和分组数据
        video_data = MdpeDetectionDataset._load_all_video_data(feature_root, label_root)
        person_videos = MdpeDetectionDataset._group_by_person(video_data)
        
        # 生成训练集和测试集
        train_list = []
        test_list = []
        
        train_stats = {'truth': 0, 'lie': 0}
        test_stats = {'truth': 0, 'lie': 0}
        
        for person_id, videos in person_videos.items():
            label_0_videos = videos['label_0'].copy()
            label_1_videos = videos['label_1'].copy()
            
            # Shuffle
            random.shuffle(label_0_videos)
            random.shuffle(label_1_videos)
            
            # 测试集 ([:3] 和 [:2])
            test_0 = label_0_videos[:3]
            test_1 = label_1_videos[:2]
            test_list.extend(test_0)
            test_list.extend(test_1)
            test_stats['truth'] += len(test_0)
            test_stats['lie'] += len(test_1)
            
            # 训练集 ([3:] 和 [2:])
            train_0 = label_0_videos[3:]
            train_1 = label_1_videos[2:]
            train_list.extend(train_0)
            train_list.extend(train_1)
            train_stats['truth'] += len(train_0)
            train_stats['lie'] += len(train_1)
        
        print(f" Train: {len(train_list)} (Truth: {train_stats['truth']}, Lie: {train_stats['lie']})")
        print(f" Test:  {len(test_list)} (Truth: {test_stats['truth']}, Lie: {test_stats['lie']})")
        
        # 创建数据集实例
        train_dataset = MdpeDetectionDataset(train_list, partial_rate=partial_rate)
        test_dataset = MdpeDetectionDataset(test_list, partial_rate=0.0)
        
        return train_dataset, test_dataset

