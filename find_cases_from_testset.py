"""
从测试集中自动找出TP/FP/FN案例
遍历测试集，根据模型预测结果自动识别不同类型的案例
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os

from models.photo import LieDetection
from configs.dolos import Args
from datasets.dolos import DOLOSDataset


def load_model(model_path, args, device):
    """加载模型"""
    model = LieDetection(args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_testset(model, dataset, device, output_dir='data_for_plot/dolos_cases'):
    """
    遍历测试集，找出TP/FP/FN案例
    
    Returns:
        cases: dict with keys 'TP', 'FP', 'FN', each containing list of case info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cases = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
    
    print(f"\n开始评估测试集 ({len(dataset)} 个样本)...")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                visual_features = sample['visual_features']
                audio_features = sample['audio_features']
                label = sample['label']
                video_name = sample['video_name']
                
                # 准备输入
                visual_list = [visual_features.to(device)]
                audio_list = [audio_features.to(device) if audio_features is not None else None]
                
                # 前向传播
                outputs = model(visual_list, audio_list, bag_labels=None)
                
                # 获取预测结果
                logits = outputs['logits']
                prob_lie = F.softmax(logits, dim=1)[0, 1].item()
                pred_label = 1 if prob_lie >= 0.5 else 0
                true_label = label if isinstance(label, int) else label
                
                # 获取注意力权重
                cross_attn = outputs['cross_attention_weights'][0].squeeze().detach().cpu().numpy()
                
                # 判断案例类型
                if true_label == 1 and pred_label == 1:
                    case_type = 'TP'
                elif true_label == 0 and pred_label == 0:
                    case_type = 'TN'
                elif true_label == 0 and pred_label == 1:
                    case_type = 'FP'
                else:
                    case_type = 'FN'
                
                # 保存案例信息
                case_info = {
                    'video_name': video_name,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'prob_lie': prob_lie,
                    'attn': cross_attn,
                    'confidence': abs(prob_lie - 0.5)  # 预测置信度
                }
                
                cases[case_type].append(case_info)
                
                if idx % 50 == 0:
                    print(f"  处理进度: {idx}/{len(dataset)}")
                    
            except Exception as e:
                print(f"  ⚠️  跳过样本 {idx}: {e}")
                continue
    
    # 打印统计信息
    print("\n" + "="*60)
    print("测试集评估结果:")
    print("="*60)
    for case_type in ['TP', 'TN', 'FP', 'FN']:
        count = len(cases[case_type])
        print(f"{case_type}: {count} 个样本")
    
    total = sum(len(cases[ct]) for ct in cases)
    accuracy = (len(cases['TP']) + len(cases['TN'])) / total if total > 0 else 0
    print(f"\n准确率: {accuracy:.2%} ({len(cases['TP']) + len(cases['TN'])}/{total})")
    print("="*60)
    
    return cases


def select_representative_cases(cases, output_dir='data_for_plot/dolos_cases', show_top_n=15, min_confidence=0.3):
    """
    从每种类型中选择最具代表性的案例
    
    选择策略：
    - 优先选择视频长度最长的案例（更多segment，可视化效果更好）
    - 同时要求置信度不能太低（>= min_confidence）
    """
    selected = {}
    
    for case_type in ['TP', 'FP', 'FN']:
        if len(cases[case_type]) == 0:
            print(f"⚠️  没有找到 {case_type} 案例")
            continue
        
        # 过滤掉置信度太低的案例
        filtered_cases = [c for c in cases[case_type] if c['confidence'] >= min_confidence]
        
        if len(filtered_cases) == 0:
            print(f"⚠️  {case_type} 没有置信度>={min_confidence}的案例，使用全部案例")
            filtered_cases = cases[case_type]
        
        # 为每个案例添加视频长度信息
        for case in filtered_cases:
            case['video_length'] = len(case['attn'])
        
        # 按视频长度排序（优先选择长视频）
        sorted_cases = sorted(filtered_cases, key=lambda x: x['video_length'], reverse=True)
        
        # 显示top-N，让用户可以挑选
        top_cases = sorted_cases[:min(show_top_n, len(sorted_cases))]
        
        print(f"\n{case_type} 案例 (按视频长度排序，显示前{len(top_cases)}个):")
        for i, case in enumerate(top_cases, 1):
            print(f"  {i}. {case['video_name']}")
            print(f"     Length={case['video_length']} segments, P(Lie)={case['prob_lie']:.3f}, "
                  f"Confidence={case['confidence']:.3f}")
            print(f"     GT={'Lie' if case['true_label']==1 else 'Truth'}, "
                  f"Pred={'Lie' if case['pred_label']==1 else 'Truth'}")
        
        # 默认选择最长的视频
        best_case = top_cases[0]
        selected[case_type] = best_case
        
        # 保存注意力权重和概率到文件
        case_name = case_type.lower()
        np.save(f"{output_dir}/{case_name}_attn.npy", best_case['attn'])
        np.save(f"{output_dir}/{case_name}_prob.npy", np.array([best_case['prob_lie']]))
        
        print(f"  ✓ 已选择: {best_case['video_name']}")
        print(f"     Length={best_case['video_length']} segments, P(Lie)={best_case['prob_lie']:.3f}")
        print(f"  ✓ 已保存: {output_dir}/{case_name}_attn.npy, {case_name}_prob.npy")
    
    return selected


def save_case_config(selected_cases, output_file='data_for_plot/dolos_cases/case_config.txt'):
    """保存案例配置，方便fig7_filmstrip_only.py使用"""
    with open(output_file, 'w') as f:
        f.write("# 自动选择的案例配置\n")
        f.write("# 可以直接复制到fig7_filmstrip_only.py的load_case_data()函数中\n\n")
        f.write("case_configs = [\n")
        
        for case_type in ['TP', 'FP', 'FN']:
            if case_type not in selected_cases:
                continue
            
            case = selected_cases[case_type]
            f.write("    {\n")
            f.write(f"        'name': '{case_type.lower()}',\n")
            f.write(f"        'video': '{case['video_name']}',\n")
            f.write(f"        'true_label': {case['true_label']},  # {'Lie' if case['true_label']==1 else 'Truth'}\n")
            f.write(f"        # P(Lie)={case['prob_lie']:.3f}, Confidence={case['confidence']:.3f}\n")
            f.write("    },\n")
        
        f.write("]\n")
    
    print(f"\n✓ 案例配置已保存到: {output_file}")


def main():
    # 配置
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_PATH = '/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/photo/model_best_fold_3_98.pth.tar'
    FOLD = 3
    OUTPUT_DIR = 'data_for_plot/dolos_cases'
    
    print(f"使用设备: {device}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试折: {FOLD}")
    
    # 加载模型
    print("\n正在加载模型...")
    model = load_model(MODEL_PATH, args, device)
    print("✓ 模型加载成功")
    
    # 加载测试集
    print(f"\n正在加载 DOLOS 第 {FOLD} 折测试集...")
    test_dataset = DOLOSDataset(
        feature_root=args.feature_root,
        fold_path=args.fold_path,
        fold_index=FOLD - 1,
        split='test',
        audio_feature_root=args.audio_feature_root,
        modality=args.modality,
        audio_dim=args.audio_dim
    )
    print(f"✓ 测试集加载完成，共 {len(test_dataset)} 个样本")
    
    # 评估测试集，找出所有案例
    cases = evaluate_testset(model, test_dataset, device, OUTPUT_DIR)
    
    # 选择最具代表性的案例
    print("\n" + "="*60)
    print("选择代表性案例:")
    print("="*60)
    selected_cases = select_representative_cases(cases, OUTPUT_DIR)
    
    # 保存配置文件
    save_case_config(selected_cases, f"{OUTPUT_DIR}/case_config.txt")
    
    print("\n" + "="*60)
    print("完成！现在可以运行 fig7_filmstrip_only.py 进行可视化")
    print("="*60)


if __name__ == "__main__":
    main()
