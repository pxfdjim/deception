"""
简化版案例分析可视化 - 只显示胶片风格的帧序列
三行帧序列，用不同颜色框标记关键帧

工作原理：
1. 案例识别：根据模型预测概率P(Lie)和真实标签自动识别TP/TN/FP/FN
   - TP (绿色): 正确识别谎言
   - TN (蓝色): 正确识别真话
   - FP (红色): 误判真话为谎言
   - FN (橙色): 漏检谎言

2. 关键帧标记：根据注意力权重找出top-2的segment，用粗边框高亮
   - 粗边框 (4.5px): 注意力最高的2个segment
   - 细边框 (2.0px): 其他segment
   - 边框颜色与案例类型一致

3. 胶片风格：黑色边缘 + 白色穿孔 + 浅灰背景，帧紧贴上下边缘
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "data_for_plot/dolos_cases"
FRAMES_DIR = '/home/pengxf/emotion/dataset/DOLOS/frames'
OUTPUT_DIR = 'figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_style():
    """配置学术风格"""
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 40,
        'axes.linewidth': 1.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def identify_case_type(prob, true_label):
    """
    识别案例类型
    
    工作原理：
    1. 根据模型预测概率P(Lie)判断预测标签：
       - 如果P(Lie) >= 0.5，预测为谎言(1)
       - 如果P(Lie) < 0.5，预测为真话(0)
    
    2. 对比预测标签和真实标签，判断案例类型：
       - TP (True Positive): 真实是谎言，预测也是谎言 ✓
       - TN (True Negative): 真实是真话，预测也是真话 ✓
       - FP (False Positive): 真实是真话，但预测为谎言 ✗
       - FN (False Negative): 真实是谎言，但预测为真话 ✗
    
    Args:
        prob: 模型预测的P(Lie)概率 (0-1之间的浮点数)
        true_label: 真实标签 (0=Truth真话, 1=Lie谎言)
    
    Returns:
        case_type: 'TP', 'TN', 'FP', 'FN'
        pred_label: 预测标签 (0或1)
    
    示例：
        - prob=0.95, true_label=1 → TP (正确识别谎言)
        - prob=0.20, true_label=0 → TN (正确识别真话)
        - prob=0.80, true_label=0 → FP (误判真话为谎言)
        - prob=0.30, true_label=1 → FN (误判谎言为真话)
    """
    pred_label = 1 if prob >= 0.5 else 0
    
    if true_label == 1 and pred_label == 1:
        return 'TP', pred_label  # True Positive
    elif true_label == 0 and pred_label == 0:
        return 'TN', pred_label  # True Negative
    elif true_label == 0 and pred_label == 1:
        return 'FP', pred_label  # False Positive
    else:
        return 'FN', pred_label  # False Negative


def load_case_data():
    """
    加载案例数据并自动识别类型
    
    优先从case_config.txt读取配置，如果不存在则使用默认配置
    """
    config_file = f"{DATA_DIR}/case_config.txt"
    
    # 尝试从case_config.txt读取配置
    case_configs = []
    if Path(config_file).exists():
        print(f"✓ 从 {config_file} 读取案例配置")
        try:
            # 简单解析case_config.txt
            with open(config_file, 'r') as f:
                content = f.read()
                # 提取配置信息（简单的字符串解析）
                import re
                # 匹配每个配置块
                pattern = r"'name':\s*'(\w+)'.*?'video':\s*'([^']+)'.*?'true_label':\s*(\d+)"
                matches = re.findall(pattern, content, re.DOTALL)
                
                for name, video, true_label in matches:
                    case_configs.append({
                        'name': name,
                        'video': video,
                        'true_label': int(true_label)
                    })
                
                print(f"  成功读取 {len(case_configs)} 个案例配置")
        except Exception as e:
            print(f"⚠️  解析配置文件失败: {e}，使用默认配置")
            case_configs = []
    
    # 如果没有读取到配置，使用默认配置
    if not case_configs:
        print("使用默认案例配置")
        case_configs = [
            {
                'name': 'tp',
                'video': 'LS_WILTY_EP12_lie3',
                'true_label': 1,
            },
            {
                'name': 'fp',
                'video': 'SB_WILTY_EP29_truth7',
                'true_label': 0,
            },
            {
                'name': 'fn',
                'video': 'YW_WILTY_EP45_lie7',
                'true_label': 1,
            },
        ]
    
    cases = []
    for config in case_configs:
        try:
            # 加载注意力权重和预测概率
            attn = np.load(f"{DATA_DIR}/{config['name']}_attn.npy").squeeze()
            prob = np.load(f"{DATA_DIR}/{config['name']}_prob.npy")[0]
            
            # 自动识别案例类型
            case_type, pred_label = identify_case_type(prob, config['true_label'])
            
            # 根据案例类型设置样式
            if case_type == 'TP':
                color = '#2ca02c'  # 深蓝色 - TP用深蓝色
                title = 'True Positive'
                desc = 'Correctly identifies deceptive behavior'
            elif case_type == 'TN':
                color = '#1f77b4'  # 蓝色
                title = 'True Negative'
                desc = 'Correctly identifies truthful behavior'
            elif case_type == 'FP':
                color = '#d62728'  # 红色
                title = 'False Positive'
                desc = 'Misclassifies truth as deception'
            else:  # FN
                color = '#ff7f0e'  # 橙色
                title = 'False Negative'
                desc = 'Fails to detect deceptive behavior'
            
            cases.append({
                'type': case_type,
                'title': title,
                'video': config['video'],
                'attn': attn,
                'prob': prob,
                'true_label': config['true_label'],
                'pred_label': pred_label,
                'color': color,
                'desc': desc
            })
            
            print(f"✓ Loaded {config['name']}: {case_type} (GT={config['true_label']}, Pred={pred_label}, P(Lie)={prob:.3f})")
            
        except Exception as e:
            print(f"✗ Warning: Failed to load {config['name']}: {e}")
    
    return cases


def _read_square_image(frame_path):
    """读取并裁剪为正方形图像"""
    if frame_path is None or not Path(frame_path).exists():
        return None
    try:
        img = plt.imread(frame_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.max() > 1.5:
            img = img / 255.0

        h, w = img.shape[:2]
        side = min(h, w)
        sy = (h - side) // 2
        sx = (w - side) // 2
        img = img[sy:sy + side, sx:sx + side]
        return img
    except Exception:
        return None


def _compose_segment_thumbnail(frame_paths):
    """将多帧叠加形成胶片效果"""
    images = []
    for frame_path in frame_paths:
        img = _read_square_image(frame_path)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        return np.ones((96, 96, 3), dtype=np.float32) * 0.92

    target_h, target_w = images[0].shape[:2]
    aligned = []
    for img in images:
        if img.shape[:2] == (target_h, target_w):
            aligned.append(img)
    
    if len(aligned) == 0:
        return np.ones((96, 96, 3), dtype=np.float32) * 0.92

    while len(aligned) < 3:
        aligned.append(aligned[-1].copy())
    aligned = aligned[:3]

    # 10%偏移叠加
    shift_ratio = 0.10
    max_shift = int(target_w * shift_ratio)
    total_shift = max_shift * (len(aligned) - 1)
    canvas_w = target_w + total_shift
    canvas = np.ones((target_h, canvas_w, 3), dtype=np.float32) * 0.95

    for idx, img in enumerate(aligned):
        x_offset = idx * max_shift
        canvas[:, x_offset:x_offset + target_w, :] = img

    return np.clip(canvas, 0.0, 1.0)


def load_segment_frame_groups(video_name, frames_dir, num_segments, frames_per_segment=3):
    """加载每个segment的关键帧"""
    frames_root = Path(frames_dir) / video_name
    frame_groups = []
    
    if not frames_root.exists():
        return [[] for _ in range(num_segments)]
    
    all_frames = sorted([p for p in frames_root.iterdir() if p.suffix.lower() == '.jpg'])
    total = len(all_frames)
    
    if total == 0:
        return [[] for _ in range(num_segments)]
    
    for idx in range(num_segments):
        start_idx = idx * 16
        end_idx = min(start_idx + 15, total - 1)
        
        if end_idx < start_idx:
            frame_groups.append([])
            continue
        
        segment_length = end_idx - start_idx + 1
        if segment_length >= frames_per_segment * 4:
            indices = np.linspace(start_idx, end_idx, frames_per_segment, dtype=int).tolist()
        elif segment_length >= frames_per_segment:
            step = segment_length // frames_per_segment
            indices = [start_idx + i * step for i in range(frames_per_segment)]
        else:
            indices = list(range(start_idx, end_idx + 1))
        
        indices = sorted(set(indices))
        group = [str(all_frames[min(max(i, 0), total - 1)]) for i in indices]
        frame_groups.append(group)
    
    return frame_groups


def plot_filmstrip_only():
    """只绘制三行胶片风格的帧序列"""
    configure_style()
    cases = load_case_data()
    
    if len(cases) == 0:
        print("Error: No cases loaded!")
        return
    
    # 创建图形：三行，每行一个案例，减小行间距
    fig = plt.figure(figsize=(18, 3.5 * len(cases)))
    gs = gridspec.GridSpec(len(cases), 1, hspace=0.15)  # 减小行间距从0.25到0.15
    
    for i, case in enumerate(cases):
        ax = fig.add_subplot(gs[i])
        ax.axis('off')
        
        attn = case['attn']
        n = len(attn)
        
        # 找出注意力最高的2个segment
        top_indices = set(np.argsort(attn)[-2:].tolist() if len(attn) >= 2 else [0])
        
        # 加载帧
        frame_groups = load_segment_frame_groups(case['video'], FRAMES_DIR, n, frames_per_segment=3)
        
        # 设置坐标范围
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        
        # 绘制胶片背景
        strip_fc = "#363636"  # 胶片边缘颜色
        main_fc = "#EFEFEF"   # 胶片主体颜色
        
        # 胶片参数
        film_height = 0.75
        film_y = 0.125
        border_h = 0.12  # 上下边缘高度
        
        # 主背景
        ax.add_patch(Rectangle((0, film_y), n, film_height, 
                              fc=main_fc, ec="none", zorder=0))
        
        # 上下胶片边缘
        ax.add_patch(Rectangle((0, film_y + film_height - border_h), n, border_h, 
                              fc=strip_fc, ec="none", zorder=1))
        ax.add_patch(Rectangle((0, film_y), n, border_h, 
                              fc=strip_fc, ec="none", zorder=1))
        
        # 胶片孔
        num_holes = n * 8
        hole_gap = n / num_holes
        hole_w = hole_gap * 0.35
        hole_h = border_h * 0.35
        
        for j in range(num_holes):
            hx = j * hole_gap + (hole_gap - hole_w) / 2
            # 上排孔
            hy_top = film_y + film_height - border_h + (border_h - hole_h) / 2
            ax.add_patch(Rectangle((hx, hy_top), hole_w, hole_h, 
                                  fc="white", ec="none", zorder=2))
            # 下排孔
            hy_bot = film_y + (border_h - hole_h) / 2
            ax.add_patch(Rectangle((hx, hy_bot), hole_w, hole_h, 
                                  fc="white", ec="none", zorder=2))
        
        # 胶片外框
        ax.add_patch(Rectangle((0, film_y), n, film_height, 
                              fill=False, ec="#202020", lw=2.5, zorder=3))
        
        # 学术风格标题
        # 格式：Case Type | Model Prediction | Ground Truth
        case_type_full = {
            'TP': 'True Positive',
            'FP': 'False Positive', 
            'FN': 'False Negative',
            'TN': 'True Negative'
        }[case['type']]
        
        true_label_str = 'Deceptive' if case['true_label'] == 1 else 'Truthful'
        pred_label_str = 'Deceptive' if case['pred_label'] == 1 else 'Truthful'
        
        # 简洁的学术风格标题（不显示具体概率数字）
        title_text = f"{case_type_full}: Prediction = {pred_label_str}, Ground Truth = {true_label_str}"
        
        ax.text(n / 2, film_y + film_height + 0.04, 
               title_text,
               ha='center', va='bottom', fontsize=30, fontfamily='Times New Roman', 
               color='#2E3440')
        
        # 计算帧的精确位置（紧贴胶片边缘）
        # 帧应该填满胶片边缘之间的空间，无缝贴合
        frame_width_data = 0.90  # 每个帧的宽度（data坐标）
        frame_height_data = film_height - 2 * border_h  # 帧的高度 = 胶片高度 - 上下边缘
        frame_y_data = film_y + border_h  # 帧的y位置 = 胶片底部 + 边缘高度
        
        # 获取axes的位置信息
        bbox = ax.get_position()
        ax_width_fig = bbox.width
        ax_height_fig = bbox.height
        
        # data坐标系范围
        data_x_range = n  # x轴范围是0到n
        data_y_range = 1.0  # y轴范围是0到1
        
        # data坐标到figure坐标的转换比例
        data_to_fig_x = ax_width_fig / data_x_range
        data_to_fig_y = ax_height_fig / data_y_range
        
        # 绘制每个segment的帧
        for seg_idx in range(n):
            segment_paths = frame_groups[seg_idx] if seg_idx < len(frame_groups) else []
            img_arr = _compose_segment_thumbnail(segment_paths)
            
            # 计算帧的x位置（在每个segment内居中）
            frame_x_data = seg_idx + (1.0 - frame_width_data) / 2.0
            
            # 转换为figure坐标
            x_fig = bbox.x0 + frame_x_data * data_to_fig_x
            y_fig = bbox.y0 + frame_y_data * data_to_fig_y
            w_fig = frame_width_data * data_to_fig_x
            h_fig = frame_height_data * data_to_fig_y
            
            # 创建帧的axes并显示图像
            ax_img = fig.add_axes([x_fig, y_fig, w_fig, h_fig])
            ax_img.imshow(img_arr)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_aspect('auto')  # 允许拉伸以填满空间
            
            # 边框：高亮关键帧
            is_highlight = seg_idx in top_indices
            border_color = case['color'] if is_highlight else '#7F8C8D'
            border_width = 4.5 if is_highlight else 2.0
            
            for sp in ax_img.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(border_color)
                sp.set_linewidth(border_width)
    
    plt.savefig(f"{OUTPUT_DIR}/fig7_filmstrip_only.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/fig7_filmstrip_only.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_DIR}/fig7_filmstrip_only.pdf & .png")
    plt.close()


if __name__ == "__main__":
    plot_filmstrip_only()
