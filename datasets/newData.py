"""
number_guess_v2 视频特征 Dataset / DataLoader
支持:
  1. 逐题训练 (NumberGuessDataset + get_fold_loaders)
  2. 按被试聚合评估 Top-K (SubjectDataset + compute_topk_accuracy)

特征: .npy (N, 768), 标签: label.csv, 五折: fold_split_new.csv
音频特征: wavlm_large_features/{subject}/{qid}.npy (WavLM Large, 1024维)
支持模态选择: 'visual' / 'audio' / 'both'
"""
from __future__ import annotations
import csv, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.audio_utils import load_audio_feature

# ======================== 路径配置 ========================
DATASET_ROOT = Path("/home/deception/dataset/number_guess_v2")
FEATURE_DIR  = DATASET_ROOT / "features" / "VideoMAEv2-Base_avg_128_stride16"
LABEL_DIR    = DATASET_ROOT / "dataset"
FOLD_CSV     = DATASET_ROOT / "fold_split_new.csv"
# =========================================================

BASELINE_QIDS = {1, 2}
TARGET_QIDS   = set(range(3, 23))  # Q3-Q22, 共20题

QUESTION_NUMBER_MAPPING = {
    3: 4,  4: 8,  5: 3,  6: 1,  7: 6,
    8: 9,  9: 2,  10: 5, 11: 7, 12: 10,
    13: 7, 14: 2, 15: 9, 16: 3, 17: 5,
    18: 1, 19: 8, 20: 10, 21: 6, 22: 4,
}


def load_fold_split() -> Dict[str, int]:
    split = {}
    with open(FOLD_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split[row["id"].strip()] = int(row["fold"])
    return split


def load_labels(subj_name: str) -> Dict[int, int]:
    label_path = LABEL_DIR / subj_name / "label.csv"
    labels = {}
    if label_path.exists():
        with open(label_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row["question_id"])] = int(row["label"])
    return labels


# ============================================================
#  方案 A-1: 逐题 Dataset (用于训练)
# ============================================================
class NumberGuessDataset(Dataset):
    """每个样本 = 一个被试的一个问题，支持视觉+音频多模态
    
    样本枚举基于 fold_split_new.csv 中的被试 + TARGET_QIDS (3-22)，
    而非扫描视觉特征目录，保证只加载划分文件中的数据。
    """
    def __init__(self, subjects, feature_dir=FEATURE_DIR,
                 include_baseline=False, target_only=True, fold_map=None,
                 audio_feature_dir=None, modality='visual', audio_dim=1024):
        self.feature_dir = Path(feature_dir)
        self.modality = modality
        self.audio_dim = audio_dim
        self.audio_feature_dir = Path(audio_feature_dir) if audio_feature_dir else None
        self.samples = []
        
        # 确定要加载的 qid 范围
        qids_to_load = sorted(TARGET_QIDS) if target_only else sorted(TARGET_QIDS | BASELINE_QIDS)
        
        for subj in subjects:
            labels = load_labels(subj)
            fold_id = fold_map.get(subj, -1) if fold_map else -1
            
            for qid in qids_to_load:
                # 根据模态检查特征文件是否存在
                visual_path = self.feature_dir / subj / f"{qid}.npy"
                audio_path = self.audio_feature_dir / subj / f"{qid}.npy" if self.audio_feature_dir else None
                
                if self.modality == 'visual':
                    if not visual_path.exists():
                        continue
                elif self.modality == 'audio':
                    if audio_path is None or not audio_path.exists():
                        continue
                elif self.modality == 'both':
                    # 双模态：视觉必须存在，音频缺失时 __getitem__ 会补零
                    if not visual_path.exists():
                        continue
                
                label = labels.get(qid, 0)
                meta = {"subj": subj, "qid": qid,
                        "number": QUESTION_NUMBER_MAPPING.get(qid), "fold": fold_id}
                self.samples.append((visual_path, label, meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label, meta = self.samples[idx]
        subj = meta['subj']
        qid = meta['qid']
        
        # ====== 加载视觉特征 ======
        visual_features = None
        if self.modality in ('visual', 'both'):
            try:
                visual_features = torch.from_numpy(np.load(npy_path)).float()
            except Exception as e:
                # print(f"❌ 加载视觉特征失败 {npy_path}: {e}")
                visual_features = torch.zeros(1, 768)
        
        # ====== 加载音频特征 ======
        audio_features = None
        if self.modality in ('audio', 'both') and self.audio_feature_dir:
            audio_path = self.audio_feature_dir / subj / f"{qid}.npy"
            audio_features = load_audio_feature(audio_path, expected_dim=self.audio_dim)
            if audio_features is None:
                audio_features = torch.zeros(1, self.audio_dim)
        
        label = torch.tensor(label, dtype=torch.long)
        return {
            'visual_features': visual_features,
            'audio_features': audio_features,
            'label': label,
            'meta': meta
        }

    def get_label_distribution(self):
        return dict(Counter(s[1] for s in self.samples))


# ============================================================
#  方案 A-2: 按被试 Dataset (用于 Top-K 评估)
# ============================================================
class SubjectDataset(Dataset):
    """
    每个样本 = 一个被试的全部20个target问题
    返回:
        visual_feats:  (20, max_seg_v, 768) — pad到该被试内最大视觉段数
        audio_feats:   (20, max_seg_a, 1024) — pad到该被试内最大音频段数
        labels: (20,)               — 其中恰好2个为1
        qids:   (20,)               — 对应的question id
        meta:   dict
    """
    def __init__(self, subjects, feature_dir=FEATURE_DIR, fold_map=None,
                 audio_feature_dir=None, modality='visual', audio_dim=1024):
        self.feature_dir = Path(feature_dir)
        self.modality = modality
        self.audio_dim = audio_dim
        self.audio_feature_dir = Path(audio_feature_dir) if audio_feature_dir else None
        self.subjects = []

        for subj in subjects:
            subj_feat_dir = self.feature_dir / subj
            if not subj_feat_dir.exists():
                continue
            labels = load_labels(subj)
            fold_id = fold_map.get(subj, -1) if fold_map else -1

            questions = []
            for qid in sorted(TARGET_QIDS):
                npy_path = subj_feat_dir / f"{qid}.npy"
                if not npy_path.exists():
                    continue
                questions.append((npy_path, qid, labels.get(qid, 0)))

            if questions:
                self.subjects.append((subj, questions, fold_id))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj, questions, fold_id = self.subjects[idx]

        v_feats_list, a_feats_list, labels_list, qids_list = [], [], [], []
        for npy_path, qid, label in questions:
            # 视觉特征
            feat = np.load(npy_path)           # (n_seg, 768)
            v_feats_list.append(torch.from_numpy(feat).float())
            
            # 音频特征
            if self.modality in ('audio', 'both') and self.audio_feature_dir:
                audio_path = self.audio_feature_dir / subj / f"{qid}.npy"
                audio_feat = load_audio_feature(audio_path, expected_dim=self.audio_dim)
                if audio_feat is None:
                    audio_feat = torch.zeros(1, self.audio_dim)
                a_feats_list.append(audio_feat)
            else:
                a_feats_list.append(torch.zeros(1, self.audio_dim))
            
            labels_list.append(label)
            qids_list.append(qid)

        n_q = len(v_feats_list)

        # pad 视觉特征
        max_seg_v = max(f.shape[0] for f in v_feats_list)
        v_padded = torch.zeros(n_q, max_seg_v, v_feats_list[0].shape[1])
        for i, f in enumerate(v_feats_list):
            v_padded[i, :f.shape[0], :] = f

        # pad 音频特征
        max_seg_a = max(f.shape[0] for f in a_feats_list)
        a_padded = torch.zeros(n_q, max_seg_a, self.audio_dim)
        for i, f in enumerate(a_feats_list):
            a_padded[i, :f.shape[0], :] = f

        labels = torch.tensor(labels_list, dtype=torch.long)
        qids = torch.tensor(qids_list, dtype=torch.long)

        meta = {"subj": subj, "fold": fold_id, "n_questions": n_q}
        return v_padded, a_padded, labels, qids, meta


def subject_collate_fn(batch):
    """按被试 collate, pad 到 batch 内最大问题数和段数"""
    B = len(batch)
    max_q     = max(b[0].shape[0] for b in batch)
    max_seg_v = max(b[0].shape[1] for b in batch)
    max_seg_a = max(b[1].shape[1] for b in batch)
    v_dim     = batch[0][0].shape[2]
    a_dim     = batch[0][1].shape[2]

    v_feats = torch.zeros(B, max_q, max_seg_v, v_dim)
    a_feats = torch.zeros(B, max_q, max_seg_a, a_dim)
    labels  = torch.full((B, max_q), -1, dtype=torch.long)
    qids    = torch.zeros(B, max_q, dtype=torch.long)
    metas   = []

    for i, (vf, af, l, q, m) in enumerate(batch):
        nq = vf.shape[0]
        v_feats[i, :nq, :vf.shape[1], :] = vf
        a_feats[i, :nq, :af.shape[1], :] = af
        labels[i, :nq] = l
        qids[i, :nq] = q
        metas.append(m)

    return v_feats, a_feats, labels, qids, metas


# ============================================================
#  Top-K 准确率计算
# ============================================================
def compute_topk_accuracy(model, loader, device, ks=(1, 2, 3)):
    """
    计算 Top-K 准确率:
      对每个被试, 模型对20题打分, 取分数最高的K题,
      如果其中包含至少1个 label=1 的题, 则该被试算"命中"
    
    Args:
        model:  接收 (visual_feats, audio_feats) 两个参数, 输出 (B, 2) 或 (B,) 分数
        loader: SubjectDataset 的 DataLoader
        ks:     要计算的 K 值
    
    Returns:
        dict: {k: accuracy} 如 {1: 0.65, 2: 0.82, 3: 0.91}
    """
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0

    with torch.no_grad():
        for v_feats, a_feats, labels, qids, metas in loader:
            # v_feats: (B, max_q, max_seg_v, v_dim)
            # a_feats: (B, max_q, max_seg_a, a_dim)
            # labels: (B, max_q), -1 = padding
            B = v_feats.shape[0]

            for i in range(B):
                valid_mask = labels[i] != -1
                valid_v = v_feats[i][valid_mask].to(device)   # (n_q, max_seg_v, v_dim)
                valid_a = a_feats[i][valid_mask].to(device)   # (n_q, max_seg_a, a_dim)
                valid_labels = labels[i][valid_mask]           # (n_q,)

                if valid_labels.sum() == 0:
                    continue

                scores = model(valid_v, valid_a)              # (n_q, 2) or (n_q,)
                if scores.dim() == 2:
                    scores = scores[:, 1]
                scores = scores.cpu()

                ranked_indices = scores.argsort(descending=True)
                ranked_labels = valid_labels[ranked_indices]

                total += 1
                for k in ks:
                    if ranked_labels[:k].sum() > 0:
                        hits[k] += 1

    results = {k: hits[k] / max(total, 1) for k in ks}
    return results


# ============================================================
#  DataLoader 工厂函数
# ============================================================
def question_collate_fn(batch):
    """
    NumberGuess collate函数 — 返回 4-tuple: (visual_list, audio_list, labels, metas)
    visual_list/audio_list 中的元素可能为 None（当该模态未启用时）
    """
    visual_list = [b['visual_features'] for b in batch]
    audio_list = [b['audio_features'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    metas = [b['meta'] for b in batch]
    return visual_list, audio_list, labels, metas


def get_fold_loaders(fold, batch_size=32, num_workers=4,
                     feature_dir=FEATURE_DIR, shuffle_train=True):
    """返回 (train_loader, val_loader) — 逐题, 用于训练"""
    fold_map = load_fold_split()
    train_subjs = [s for s, f in fold_map.items() if f != fold]
    val_subjs   = [s for s, f in fold_map.items() if f == fold]

    train_ds = NumberGuessDataset(train_subjs, feature_dir, fold_map=fold_map)
    val_ds   = NumberGuessDataset(val_subjs, feature_dir, fold_map=fold_map)

    print(f"Fold {fold}: train={len(train_ds)} ({len(train_subjs)} subjs), "
          f"val={len(val_ds)} ({len(val_subjs)} subjs)")
    print(f"  Train: {train_ds.get_label_distribution()}, "
          f"Val: {val_ds.get_label_distribution()}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, collate_fn=question_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=question_collate_fn, pin_memory=True)
    return train_loader, val_loader


def get_subject_loader(fold, batch_size=8, num_workers=4,
                       feature_dir=FEATURE_DIR, split="val",
                       audio_feature_dir=None, modality='visual', audio_dim=1024):
    """返回 SubjectDataset 的 DataLoader — 按被试, 用于 Top-K 评估"""
    fold_map = load_fold_split()
    if split == "val":
        subjs = [s for s, f in fold_map.items() if f == fold]
    elif split == "train":
        subjs = [s for s, f in fold_map.items() if f != fold]
    else:
        subjs = list(fold_map.keys())

    ds = SubjectDataset(subjs, feature_dir, fold_map=fold_map,
                        audio_feature_dir=audio_feature_dir,
                        modality=modality, audio_dim=audio_dim)
    print(f"SubjectDataset ({split}): {len(ds)} subjects")
    return DataLoader(ds, batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=subject_collate_fn, pin_memory=True)


# ============================================================
if __name__ == "__main__":
    # 1. 测试逐题 DataLoader
    print("=== 逐题 DataLoader (训练用) ===")
    for fold in range(1, 6):
        train_loader, val_loader = get_fold_loaders(fold=fold, batch_size=64)
        feats, labels, metas = next(iter(train_loader))
        print(f"  batch: feats={feats.shape}, labels={labels.shape}, "
              f"lie_ratio={labels.float().mean():.3f}\n")

    # 2. 测试被试级 DataLoader
    print("\n=== 被试级 DataLoader (Top-K评估用) ===")
    subj_loader = get_subject_loader(fold=1, batch_size=4)
    feats, labels, qids, metas = next(iter(subj_loader))
    print(f"  feats={feats.shape}")          # (B, 20, max_seg, 768)
    print(f"  labels={labels.shape}")         # (B, 20)
    print(f"  每个被试lie数: {[int(l[l>=0].sum()) for l in labels]}")
    print(f"  被试: {[m['subj'] for m in metas]}")

    # 3. 模拟 Top-K 评估 (用随机分数)
    print("\n=== 模拟 Top-K (随机baseline) ===")
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0])  # 随机分数

    dummy = DummyModel()
    topk = compute_topk_accuracy(dummy, subj_loader, "cpu", ks=(1, 2, 3))
    print(f"  随机baseline: Top1={topk[1]:.3f}, Top2={topk[2]:.3f}, Top3={topk[3]:.3f}")
    print(f"  (理论值: Top1≈{2/20:.3f}, Top2≈{1-(18*17)/(20*19):.3f}, Top3≈{1-(18*17*16)/(20*19*18):.3f})")