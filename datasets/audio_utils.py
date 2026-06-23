# -*- coding: utf-8 -*-
"""
音频特征加载工具函数
用于加载 WavLM Large 音频特征
"""

import torch
import numpy as np
from pathlib import Path


def load_audio_feature(audio_path, expected_dim=1024):
    """
    加载音频特征文件 (.npy)
    
    Args:
        audio_path: 音频特征文件路径
        expected_dim: 期望的特征维度 (WavLM Large = 1024)
    
    Returns:
        torch.Tensor: 音频特征 [T, D] 或 None（加载失败时）
    """
    try:
        audio = np.load(str(audio_path))
        audio = torch.from_numpy(audio).float()
        
        # 处理不同维度情况
        if audio.ndim == 1:
            # [D] -> [1, D]  单帧/已平均
            audio = audio.unsqueeze(0)
        elif audio.ndim == 3:
            # [1, T, D] -> [T, D]  去掉batch维度
            audio = audio.squeeze(0)
        
        # 验证特征维度
        if audio.ndim == 2 and audio.shape[1] != expected_dim:
            print(f"⚠️  音频特征维度不匹配: 期望 {expected_dim}, 实际 {audio.shape[1]}, 文件: {audio_path}")
        
        return audio
    except Exception as e:
        print(f"❌ 加载音频特征失败 {audio_path}: {e}")
        return None
