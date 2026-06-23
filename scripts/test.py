# import sqlite3

# conn = sqlite3.connect("/home/pengxf/work/TDD/Video_MAEV2/Deception/pplg_tuning.db")
# cursor = conn.cursor()

# # 列出所有表
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
# print("Tables:", tables)

# # 查询某个表的前几条记录
# cursor.execute("SELECT * FROM your_table LIMIT 5;")
# rows = cursor.fetchall()
# for row in rows:
#     print(row)

# conn.close()
# /home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_128frames/005/005_01.pt
# import torch

# path = "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames/005/005_02.pt"

# # 加载文件
# data = torch.load(path, map_location="cpu",weights_only=True)

# # 如果是 tensor
# if isinstance(data, torch.Tensor):
#     print("Tensor shape:", data.shape)

# # 如果是 dict 或其他结构，尝试打印里面的键和每个 tensor 的形状
# elif isinstance(data, dict):
#     for k, v in data.items():
#         if isinstance(v, torch.Tensor):
#             print(f"Key: {k}, Shape: {v.shape}")
#         else:
#             print(f"Key: {k}, Type: {type(v)}")

# else:
#     print("Loaded object type:", type(data))
# import torch
# from pathlib import Path
# from collections import Counter

# # 数据路径
# base_path = Path("/home/pengxf/work/TDD/Video_MAEV2/data/")

# # 初始化统计容器
# n_counter = Counter()
# total_samples = 0

# # 遍历每个受试者文件夹
# for subject_dir in base_path.iterdir():
#     if subject_dir.is_dir():
#         # 遍历该受试者下的所有 .pt 文件
#         for pt_file in subject_dir.glob("*.pt"):
#             try:
#                 data = torch.load(pt_file, map_location="cpu", weights_only=True)
                
#                 # 如果是 tensor，直接统计 n
#                 if isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[1] == 768:
#                     n_counter[data.shape[0]] += 1
#                     total_samples += 1

#                 # 如果是 dict，尝试遍历里面的 tensor
#                 elif isinstance(data, dict):
#                     for v in data.values():
#                         if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == 768:
#                             n_counter[v.shape[0]] += 1
#                             total_samples += 1

#             except Exception as e:
#                 print(f"Error loading {pt_file}: {e}")

# # 打印统计结果
# print("Feature n distribution (for n=1~8):")
# for i in range(1, 9):
#     print(f"n={i}: {n_counter.get(i, 0)} samples")

# print(f"\nTotal number of feature samples: {total_samples}")

import torch
from pathlib import Path

# --- 请在这里修改为您自己的配置 ---

# 1. 特征文件所在的根目录
#    (即包含 "001", "002", ... 这些子文件夹的目录)
FEATURE_ROOT_PATH = "/home/pengxf/emotion/dataset/SEUMLD/features/videomaev2_raw_128frames"

# 2. 您想要检查的受试者ID (请确保是字符串格式)
SUBJECT_ID_TO_INSPECT = "020" # 例如，检查ID为 "005" 的受试者

# ------------------------------------

def inspect_subject_features(feature_root, subject_id):
    """
    加载指定受试者的所有特征文件，打印每个文件的形状，
    然后将它们叠加并打印最终形状。
    """
    subject_dir = Path(feature_root) / subject_id
    
    print("="*60)
    print(f"正在检查受试者: {subject_id}")
    print(f"特征文件夹路径: {subject_dir}")
    print("="*60)

    if not subject_dir.is_dir():
        print(f"错误：找不到该受试者的文件夹。请检查路径和ID是否正确。")
        return

    # 查找该目录下所有的 .pt 特征文件
    feature_files = sorted(list(subject_dir.glob("*.pt")))

    if not feature_files:
        print(f"未在该文件夹下找到任何 .pt 特征文件。")
        return

    all_feature_tensors = []
    total_instances = 0

    print("\n--- 发现的每个问题（特征文件）的形状 ---")
    for i, file_path in enumerate(feature_files):
        try:
            # 加载特征张量
            feature_tensor = torch.load(file_path, map_location='cpu')
            
            # 检查是否是有效的张量
            if not isinstance(feature_tensor, torch.Tensor) or feature_tensor.ndim != 2:
                print(f"文件 {file_path.name} 内容格式不正确，已跳过。")
                continue

            shape = feature_tensor.shape
            print(f"文件 {i+1:02d} ({file_path.name}):\t 形状 = {shape}")
            
            all_feature_tensors.append(feature_tensor)
            total_instances += shape[0]

        except Exception as e:
            print(f"加载文件 {file_path.name} 时出错: {e}")
    
    print("\n" + "-"*60)

    if not all_feature_tensors:
        print("没有成功加载任何有效的特征张量。")
        return
        
    # --- 将所有加载的特征张量进行重合（叠加）---
    try:
        combined_tensor = torch.cat(all_feature_tensors, dim=0)
        
        print("\n--- 所有特征重合后的最终结果 ---")
        print(f"成功合并了 {len(all_feature_tensors)} 个特征文件。")
        print(f"所有文件的实例数总和: {total_instances}")
        print(f"最终叠加后的张量形状: {combined_tensor.shape}")
        
        # 验证
        if combined_tensor.shape[0] == total_instances:
            print("✅ 验证成功：叠加后的总实例数与各文件实例数之和相符。")
        else:
            print("❌ 验证失败：叠加后的实例数与预期不符，请检查代码。")

    except Exception as e:
        print(f"叠加张量时出错: {e}")

    print("="*60)


if __name__ == "__main__":
    inspect_subject_features(FEATURE_ROOT_PATH, SUBJECT_ID_TO_INSPECT)