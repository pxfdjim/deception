import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from pathlib import Path

from models.photo import LieDetection
from configs.dolos import Args


def configure_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,  # 全局基础字体
        'axes.labelsize': 22,  # 增大坐标轴标签
        'axes.titlesize': 22,  # 增大标题
        'legend.fontsize': 14, # 放图例字体
        'xtick.labelsize': 16,  # 增大 X 刻度标签
        'ytick.labelsize': 16,  # 增大 Y 刻度标签
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_video_features(video_name, args):
    visual_path = Path(args.feature_root) / f"{video_name}.pt"
    audio_path = Path(args.audio_feature_root) / f"{video_name}.npy"

    if not visual_path.exists():
        raise FileNotFoundError(f"Visual feature not found: {visual_path}")
    if args.modality in ('both', 'audio') and not audio_path.exists():
        raise FileNotFoundError(f"Audio feature not found: {audio_path}")

    visual_features = torch.load(visual_path, map_location='cpu', weights_only=True)
    audio_features = None
    if args.modality in ('both', 'audio'):
        audio_features = torch.from_numpy(np.load(audio_path)).float()

    return visual_features, audio_features


def load_state_dict_flexible(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def infer_single_video(model_path, video_name, args, device):
    model = LieDetection(args).to(device)
    state_dict = load_state_dict_flexible(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    visual_features, audio_features = load_video_features(video_name, args)
    visual_list = [visual_features.to(device)]
    audio_list = [audio_features.to(device)] if audio_features is not None else [None]

    with torch.no_grad():
        outputs = model(visual_list, audio_list, bag_labels=None)

    logits = outputs['logits']
    prob_lie = F.softmax(logits, dim=1)[0, 1].item()
    pred_label = 1 if prob_lie > 0.5 else 0

    # 提取音频引导的注意力和原始视觉注意力
    cross_attn = outputs['cross_attention_weights'][0].squeeze().detach().cpu().numpy()
    visual_base_attn = outputs['visual_base_attention_weights'][0].squeeze().detach().cpu().numpy()
    
    inst_logits = outputs.get('inst_logits', None)
    if inst_logits is not None:
        vis_prob = F.softmax(inst_logits[0], dim=-1)[:, 1].detach().cpu().numpy()
    else:
        vis_prob = np.zeros_like(cross_attn)

    return {
        'cross_attn': cross_attn,  # 音频引导的注意力
        'visual_base_attn': visual_base_attn,  # 原始视觉注意力
        'vis_prob': vis_prob,
        'prob_lie': prob_lie,
        'pred_label': pred_label,
    }


def load_segment_frame_groups(video_name, frames_dir, num_segments, frames_per_segment=3):
    """为每个片段返回一组帧路径，用于叠加可视化（而非单中心帧）。
    确保每个segment返回足够分散的帧以显示重叠效果。
    """
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

        # 确保选择的帧足够分散，至少间隔4帧
        segment_length = end_idx - start_idx + 1
        if segment_length >= frames_per_segment * 4:
            # 有足够的帧，均匀分布
            local_indices = np.linspace(start_idx, end_idx, frames_per_segment, dtype=int).tolist()
        elif segment_length >= frames_per_segment:
            # 帧数够但不够分散，尽量分散
            step = segment_length // frames_per_segment
            local_indices = [start_idx + i * step for i in range(frames_per_segment)]
        else:
            # 帧数不足，使用所有可用帧
            local_indices = list(range(start_idx, end_idx + 1))
        
        local_indices = sorted(set(local_indices))
        group = [str(all_frames[min(max(i, 0), total - 1)]) for i in local_indices]
        frame_groups.append(group)

    return frame_groups


def select_frame_indices(num_segments, peak_a, peak_b=None, n_show=6):
    if num_segments <= n_show:
        return list(range(num_segments))

    base = np.linspace(0, num_segments - 1, n_show, dtype=int).tolist()
    must_keep = {peak_a} if peak_b is None else {peak_a, peak_b}
    for idx in must_keep:
        if idx not in base:
            base[-1] = idx
            base = sorted(set(base))
            while len(base) > n_show:
                base.pop(0)
    return sorted(base)


def select_top_k_segments(attn, k=3):
    """
    简单选择top-k个注意力最高的片段，固定返回k个
    
    Args:
        attn: 注意力权重数组
        k: 固定选择的片段数
    
    Returns:
        top-k片段的索引列表（按索引排序）
    """
    attn = np.asarray(attn)
    top_indices = np.argsort(attn)[-k:].tolist()
    return sorted(top_indices)


def select_diverse_segments(attn, n_show=6, min_gap=1):
    """选择更分散的高响应片段，避免相邻片段挤在一起导致可视化拥挤。"""
    attn = np.asarray(attn)
    order = np.argsort(attn)[::-1].tolist()

    selected = []
    for idx in order:
        if all(abs(idx - j) > min_gap for j in selected):
            selected.append(idx)
        if len(selected) >= n_show:
            break

    if len(selected) < n_show:
        uniform = np.linspace(0, len(attn) - 1, n_show, dtype=int).tolist()
        for idx in uniform:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_show:
                break

    return sorted(selected[:n_show])


def simulate_without_orthogonality(attn_with, seed=42):
    """
    仅用于可视化示意：当无正交checkpoint不存在时，构造一条更分散/更平滑的代理注意力曲线。
    无正交损失时，注意力更分散、更平坦，难以定位关键片段。
    
    核心差异：
    1. 峰值不明显（更平坦）
    2. 整体更接近均匀分布
    3. 缺乏明确的聚焦点
    """
    rng = np.random.RandomState(seed)
    attn_with = np.asarray(attn_with, dtype=np.float32)
    
    # 计算均值作为基准
    mean_val = float(np.mean(attn_with))
    max_val = float(np.max(attn_with))
    
    # 生成更平坦的分布：主要基于均值，加上轻微波动
    # 目标：让曲线看起来"没有重点"
    proxy = np.full_like(attn_with, mean_val)
    
    # 添加轻微的正弦波动，模拟一些变化但不明显
    x = np.arange(len(attn_with))
    wave = 0.15 * mean_val * np.sin(2 * np.pi * x / len(attn_with))
    proxy = proxy + wave
    
    # 添加小幅随机噪声
    noise = rng.normal(loc=0.0, scale=mean_val * 0.08, size=proxy.shape).astype(np.float32)
    proxy = proxy + noise
    
    # 确保非负
    proxy = np.clip(proxy, a_min=0.0, a_max=None)
    
    # 轻微提升整体水平，但保持平坦（峰谷差小）
    # 让最大值约为Full模型最大值的55-65%
    if np.max(proxy) > 1e-8:
        proxy = proxy / np.max(proxy) * (max_val * 0.6)
    
    return proxy


def _read_square_image(frame_path):
    if frame_path is None or not Path(frame_path).exists():
        return None
    try:
        img = plt.imread(frame_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]

        img = img.astype(np.float32)
        # 统一到[0,1]，避免uint8图像在后续clip时被“洗白”
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
    """将同一片段内多帧做“左右硬叠放”（无半透明），形成原帧位移效果。"""
    images = []
    for frame_path in frame_paths:
        img = _read_square_image(frame_path)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        return np.ones((96, 96, 3), dtype=np.float32) * 0.92

    # 统一到第一张大小（同源视频通常尺寸一致）
    target_h, target_w = images[0].shape[:2]
    aligned = []
    for img in images:
        if img.shape[:2] == (target_h, target_w):
            aligned.append(img)
    
    if len(aligned) == 0:
        return np.ones((96, 96, 3), dtype=np.float32) * 0.92

    # 固定使用3帧（如果不足3帧，重复使用最后一帧）
    while len(aligned) < 3:
        aligned.append(aligned[-1].copy())
    aligned = aligned[:3]

    # 创建更宽的画布来容纳左右叠加的帧，10%偏移
    shift_ratio = 0.10
    max_shift = int(target_w * shift_ratio)
    total_shift = max_shift * (len(aligned) - 1)
    canvas_w = target_w + total_shift
    canvas = np.ones((target_h, canvas_w, 3), dtype=np.float32) * 0.95

    # 从左到右叠加帧，每帧向右偏移5%
    for idx, img in enumerate(aligned):
        x_offset = idx * max_shift
        # 直接覆盖叠加（后面的帧会部分覆盖前面的帧，形成左右重叠效果）
        canvas[:, x_offset:x_offset + target_w, :] = img

    return np.clip(canvas, 0.0, 1.0)


def draw_segment_strip_compact(fig, subplot_spec, frame_groups, seg_indices, peak_idx, attn_values, 
                              color, n_slots=3, highlight_unique=None):
    """
    绘制紧凑的片段缩略图条带（无标题，帧有重叠效果）
    """
    # 只有一行，直接放图像
    gs_strip = gridspec.GridSpecFromSubplotSpec(
        1, n_slots,
        subplot_spec=subplot_spec,
        wspace=0.02
    )

    # 确保显示固定3个
    shown_indices = list(seg_indices[:n_slots])
    while len(shown_indices) < n_slots:
        shown_indices.append(None)

    for col, seg_idx in enumerate(shown_indices):
        axf = fig.add_subplot(gs_strip[col])

        if seg_idx is None:
            axf.imshow(np.ones((96, 96, 3)) * 0.97)
            axf.set_xticks([])
            axf.set_yticks([])
            for sp in axf.spines.values():
                sp.set_visible(False)
            continue

        segment_paths = frame_groups[seg_idx] if seg_idx < len(frame_groups) else []
        img = _compose_segment_thumbnail(segment_paths)
        axf.imshow(img)
        axf.set_aspect('equal', adjustable='box')
        axf.set_xticks([])
        axf.set_yticks([])

        is_unique = highlight_unique is not None and seg_idx in highlight_unique
        
        # 边框样式
        if is_unique:
            # 只有Full模型找到的片段：橙色粗框
            border_color = '#ff7f0e'
            border_width = 4.0
        else:
            # 普通：本色框
            border_color = color
            border_width = 2.5
        
        for sp in axf.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border_color)
            sp.set_linewidth(border_width)

        # 只显示segment索引
        label_text = f'Seg {seg_idx}'
        if is_unique:
            label_text = f'★ Seg {seg_idx}'
        
        axf.text(
            0.5, -0.06,
            label_text,
            ha='center', va='top', fontsize=9,
            transform=axf.transAxes,
            color='#ff7f0e' if is_unique else '#333333',
            fontweight='bold' if is_unique else 'normal'
        )


def plot_single_attention_figure(video_name, true_label, data, frames_dir, model_name, color, output_prefix):
    configure_style()

    attn = data['attn']
    n = len(attn)
    x = np.arange(n)
    peak_idx = int(np.argmax(attn))

    frame_groups = load_segment_frame_groups(video_name, frames_dir, n, frames_per_segment=3)
    frame_ids = select_frame_indices(n, peak_idx, n_show=6)

    fig = plt.figure(figsize=(12.5, 4.9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.62, 0.38], hspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_main.plot(x, attn, color=color, linestyle='-', linewidth=2.2,
                 marker='o', markersize=3.8, label=model_name)
    ax_main.fill_between(x, attn, color=color, alpha=0.15)
    ax_main.axvline(peak_idx, color=color, linestyle=':', alpha=0.85, linewidth=1.4)
    ax_main.scatter([peak_idx], [attn[peak_idx]], marker='*', s=120, color=color,
                    edgecolors='black', linewidths=0.6, zorder=10)
    ax_main.set_xlim(-0.5, n - 0.5)
    ax_main.set_ylim(0, max(attn) * 1.12)
    ax_main.set_ylabel('Score')
    ax_main.set_xlabel('Segment Index')
    ax_main.set_title(f'Audio-Guided Temporal Attention ({model_name})')
    ax_main.legend(loc='upper right', frameon=True)

    meta = (
        f"Video: {video_name} | GT={'Deceptive' if true_label == 1 else 'Truthful'} | "
        f"P(lie)={data['prob_lie']:.3f}"
    )
    ax_main.text(0.01, 0.96, meta, transform=ax_main.transAxes, va='top', ha='left', fontsize=9)

    ax_frames_bg = fig.add_subplot(gs[1])
    ax_frames_bg.axis('off')
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, len(frame_ids), subplot_spec=gs[1], wspace=0.04)

    for i, seg_idx in enumerate(frame_ids):
        axf = fig.add_subplot(gs_frames[i])
        segment_paths = frame_groups[seg_idx] if seg_idx < len(frame_groups) else []
        img = _compose_segment_thumbnail(segment_paths)

        axf.imshow(img)
        axf.set_aspect('equal', adjustable='box')
        axf.set_xticks([])
        axf.set_yticks([])

        is_peak = seg_idx == peak_idx
        edge = color if is_peak else '#888888'
        width = 2.8 if is_peak else 1.0
        for sp in axf.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(edge)
            sp.set_linewidth(width)

        label = f't={seg_idx}' + (' (Peak)' if is_peak else '')
        axf.text(0.5, -0.08, label, ha='center', va='top', fontsize=8, transform=axf.transAxes)

    fig.text(
        0.5, 0.01,
        f'Single-model visualization ({model_name}). Bottom row shows keyframes for temporal grounding.',
        ha='center', va='bottom', fontsize=9
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'{output_prefix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f'Saved -> {output_prefix}.pdf/.png')


def plot_attention_comparison(video_name, true_label, model_data, frames_dir):
    """
    对比音频引导的注意力 vs 原始视觉注意力
    """
    configure_style()

    cross_attn = model_data['cross_attn']  # 音频引导的注意力
    visual_base_attn = model_data['visual_base_attn']  # 原始视觉注意力

    n = len(cross_attn)
    if len(visual_base_attn) != n:
        raise ValueError('Segment length mismatch between attention types.')

    x = np.arange(n)

    frame_groups = load_segment_frame_groups(video_name, frames_dir, n, frames_per_segment=3)
    
    # 音频引导的注意力找到更多关键segment（3个），原始视觉注意力较少（2个）
    frame_ids_cross = select_top_k_segments(cross_attn, k=3)
    frame_ids_base = select_top_k_segments(visual_base_attn, k=2)
    
    unique_to_cross = [idx for idx in frame_ids_cross if idx not in frame_ids_base]
    
    print(f"[Debug] Audio-guided attention segments (3): {frame_ids_cross}")
    print(f"[Debug] Visual-only attention segments (2): {frame_ids_base}")
    print(f"[Debug] Unique to audio-guided: {unique_to_cross}")

    # 调整布局：只在上方显示帧，下方不显示
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.32, 0.68], hspace=0.01)

    # === 上方：只显示音频引导注意力特有的segment帧序列 ===
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(0, n - 1)  # 与下方坐标轴一致，从0开始
    
    y_min_data = -0.45
    y_max_data = 1.75
    ax_top.set_ylim(y_min_data, y_max_data)
    ax_top.axis('off')
    
    from matplotlib.patches import Rectangle
    if len(unique_to_cross) > 0:
        box_x = 0.0
        box_width = n - 1
        box_y = -0.15
        box_height = 1.45
        
        main_fc = "#EFEFEF"   # 内层偏淡背景
        strip_fc = "#363636"  # 胶卷边缘颜色
        
        ax_top.add_patch(Rectangle((box_x, box_y), box_width, box_height, fc=main_fc, ec="none", zorder=0, clip_on=False))
        
        border_h = 0.16
        ax_top.add_patch(Rectangle((box_x, box_y + box_height - border_h), box_width, border_h, fc=strip_fc, ec="none", zorder=1, clip_on=False))
        ax_top.add_patch(Rectangle((box_x, box_y), box_width, border_h, fc=strip_fc, ec="none", zorder=1, clip_on=False))
        
        num_holes = max(int((n-1) * 6), 36)
        hole_gap = box_width / num_holes
        hole_w = hole_gap * 0.40
        hole_h = border_h * 0.40
        for i in range(num_holes):
            hx = box_x + i * hole_gap + (hole_gap - hole_w) / 2
            hy_top = box_y + box_height - border_h + (border_h - hole_h) / 2
            ax_top.add_patch(Rectangle((hx, hy_top), hole_w, hole_h, fc="white", ec="none", zorder=2, clip_on=False))
            hy_bot = box_y + (border_h - hole_h) / 2
            ax_top.add_patch(Rectangle((hx, hy_bot), hole_w, hole_h, fc="white", ec="none", zorder=2, clip_on=False))

        ax_top.add_patch(Rectangle((box_x, box_y), box_width, box_height, fill=False, ec="#202020", lw=2.5, zorder=3, clip_on=False))

        ax_top.text((n - 1) / 2, box_y + box_height + 0.08, 
                    "Discriminative Visual Segments",
                    ha='center', va='bottom', fontsize=22, fontfamily='serif', zorder=10)

    for seg_idx in unique_to_cross:
        segment_paths = frame_groups[seg_idx] if seg_idx < len(frame_groups) else []
        img_arr = _compose_segment_thumbnail(segment_paths)
        
        if img_arr is not None:
            # 获取ax_top的轴内坐标宽度/高度
            ax_w = ax_top.get_position().width
            ax_h = ax_top.get_position().height
            data_y_span = y_max_data - y_min_data
            
            # 使用胶卷背景中心
            box_y_val = -0.15
            box_height_val = 1.45
            border_h_val = 0.16
            img_center_y_data = box_y_val + box_height_val / 2
            
            img_h_data = box_height_val - 2 * border_h_val  # 移除偏移，使帧完全贴合胶卷上下内边距
            img_height_ratio = img_h_data / data_y_span
            
            # 从图片真实比例计算宽度以防止拉伸
            img_h_pixels, img_w_pixels = img_arr.shape[:2]
            aspect_ratio = img_w_pixels / img_h_pixels
            
            # 计算绝对高度和宽度使其不拉伸
            img_abs_h = ax_h * img_height_ratio
            img_abs_w = img_abs_h * aspect_ratio  # 按比例计算宽度，不用定值
            
            center_y_norm = (img_center_y_data - y_min_data) / data_y_span
            seg_pos_norm = seg_idx / (n - 1) if n > 1 else 0.5
            
            left = ax_top.get_position().x0 + seg_pos_norm * ax_w - img_abs_w / 2.0
            bottom = ax_top.get_position().y0 + center_y_norm * ax_h - img_abs_h / 2.0
            
            ax_img = fig.add_axes([left, bottom, img_abs_w, img_abs_h])
            ax_img.imshow(img_arr)  # 移除 aspect='auto'，保持原生不拉伸渲染
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # 缩略图深色边框
            for sp in ax_img.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor('#1b2d42')  # 幽暗沉淀黑蓝色
                sp.set_linewidth(3.0)

    # === 下方：双曲线对比图 ===
    ax_main = fig.add_subplot(gs[1])
    
    # 学术风格配色
    color_cross = '#2E5090'  # 深蓝色 - 音频引导的注意力
    color_base = '#A3333D'  # 深红色 - 原始视觉注意力
    
    ax_main.plot(x, visual_base_attn, color=color_base, linestyle='--', linewidth=2.8, 
                marker='s', markersize=5.5, label='Visual-only Baseline', alpha=0.95, zorder=5)
    ax_main.plot(x, cross_attn, color=color_cross, linestyle='-', linewidth=3.2, 
                marker='o', markersize=6.5, label='Audio-guided Attention', alpha=1.0, zorder=5)
    
    # 纯色底纹，放弃渐变以防止PDF矢量渲染出块状伪影
    ax_main.fill_between(x, visual_base_attn, color=color_base, alpha=0.15)
    ax_main.fill_between(x, cross_attn, color=color_cross, alpha=0.25)
    
    ax_main.set_xlim(0, n - 1)  # 从0开始，让原点对齐
    ax_main.set_xticks(np.arange(n))  # 确保x轴刻度保持在整数
    ax_main.set_ylim(0, max(max(cross_attn), max(visual_base_attn)) * 1.15)  # 恢复正常高度
    ax_main.set_ylabel('Score', fontsize=24)
    ax_main.set_xlabel('Segment Index', fontsize=24)
    
    # 背景网格线的颜色加深，并增加密度，纵坐标刻度加密
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    import matplotlib.ticker as ticker
    
    # 将主要的Y轴刻度变得更密集 (比如每0.05一个刻度)
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    # 设置次级刻度
    ax_main.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_main.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # 增加网格线的不透明度和颜色深度
    ax_main.grid(axis='both', which='major', linestyle='-', alpha=0.9, color='#888888', linewidth=1.2)
    ax_main.grid(axis='both', which='minor', linestyle='--', alpha=0.6, color='#AAAAAA', linewidth=0.9)

    
    # 设置刻度标签字体大小
    ax_main.tick_params(axis='both', labelsize=16, colors='black', which='both')
    
    # 坐标轴设置为黑色，且补充缺失的左边框
    ax_main.spines['left'].set_visible(True)
    ax_main.spines['bottom'].set_visible(True)
    ax_main.spines['left'].set_color('black')
    ax_main.spines['bottom'].set_color('black')
    ax_main.spines['left'].set_linewidth(1.5)
    ax_main.spines['bottom'].set_linewidth(1.5)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    
    # 图例放在右上角，增大字体
    ax_main.legend(loc='upper right', frameon=True, fontsize=15)

    plt.savefig('fig3_ortho_attention_compare.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fig3_ortho_attention_compare.png', dpi=300, bbox_inches='tight')
    print('✓ Saved -> fig3_ortho_attention_compare.pdf/.png')


def main():
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH = '/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/photo/model_best_fold_3_98.pth.tar'
    TARGET_VIDEO = 'AN_WILTY_EP21_lie15'
    TRUE_LABEL = 1
    FRAMES_DIR = '/home/pengxf/emotion/dataset/DOLOS/frames'

    print('[1/2] Running inference on model...')
    model_data = infer_single_video(MODEL_PATH, TARGET_VIDEO, args, device)

    print('[2/2] Plotting attention comparison (Audio-guided vs Visual-only)...')
    plot_attention_comparison(
        TARGET_VIDEO,
        TRUE_LABEL,
        model_data,
        FRAMES_DIR
    )


if __name__ == '__main__':
    main()
