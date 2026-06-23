"""
Figure 4: Complementarity Analysis
展示正交损失对视觉基础分支和音频引导分支互补性的影响
需要在测试集上收集多个样本的余弦相似度分布
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from models.photo import LieDetection
from datasets.dataloaderFactory import create_dolos_dataloaders
from datasets.dolos import dolos_collate_fn
from configs.dolos import Args

try:
    from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon
except Exception:
    ks_2samp = None
    mannwhitneyu = None
    wilcoxon = None


def configure_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.5,
        'grid.color': '#b0b0b0',
        'grid.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.alpha': 0.85
    })


def collect_similarity_from_testset(model, test_loader, device):
    """
    在测试集上收集余弦相似度分布
    
    Returns:
        sim_with_loss: 使用正交损失时的相似度
        sim_no_loss: 不使用正交损失时的相似度
    """
    model.eval()
    
    all_cos_sims = []
    all_cos_sims_no_loss = []
    missing_key_batches = 0
    total_batches = 0
    
    print("Collecting similarity distributions from test set...")
    
    with torch.no_grad():
        for visual_list, audio_list, bag_labels, ids in test_loader:
            total_batches += 1
            # 移到GPU
            visual_list = [v.to(device, non_blocking=True) if v is not None else None for v in visual_list]
            audio_list = [a.to(device, non_blocking=True) if a is not None else None for a in audio_list]
            
            # 前向传播
            outputs = model(visual_list, audio_list, bag_labels=None)
            
            # 收集相似度
            if 'cos_sim' in outputs:
                all_cos_sims.append(np.asarray(outputs['cos_sim'].detach().cpu().numpy()).reshape(-1))
            if 'cos_sim_no_loss' in outputs:
                all_cos_sims_no_loss.append(np.asarray(outputs['cos_sim_no_loss'].detach().cpu().numpy()).reshape(-1))
            if ('cos_sim' not in outputs) or ('cos_sim_no_loss' not in outputs):
                missing_key_batches += 1

    if len(all_cos_sims) == 0 or len(all_cos_sims_no_loss) == 0:
        raise RuntimeError(
            "No valid 'cos_sim' / 'cos_sim_no_loss' were collected from model outputs. "
            "For Fig.4 complementarity, set args.modality='both' and use a checkpoint trained/evaluated with both modalities. "
            f"(total_batches={total_batches}, missing_key_batches={missing_key_batches})"
        )

    sim_with_loss = np.concatenate(all_cos_sims, axis=0)
    sim_no_loss = np.concatenate(all_cos_sims_no_loss, axis=0)

    if sim_with_loss.size < 2 or sim_no_loss.size < 2:
        raise RuntimeError(
            "Too few similarity samples for distribution plotting. "
            f"Got n_with={sim_with_loss.size}, n_without={sim_no_loss.size}. "
            "Please verify test split size / batch sampling."
        )
    
    return sim_with_loss, sim_no_loss


def plot_complementarity(sim_with_loss, sim_no_loss):
    """兼容旧接口：默认绘制Version D（paired）"""
    plot_complementarity_variants(sim_with_loss, sim_no_loss, mode='paired')


def _cliffs_delta(x, y):
    """高效计算Cliff's Delta，避免O(n*m)内存开销。"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    y_sorted = np.sort(y)
    left = np.searchsorted(y_sorted, x, side='left')
    right = np.searchsorted(y_sorted, x, side='right')
    less_count = left
    greater_count = y.size - right
    delta = np.sum(greater_count - less_count) / (x.size * y.size)
    return float(delta)


def _compute_stats(sim_with_loss, sim_no_loss):
    x = np.asarray(sim_with_loss).ravel()
    y = np.asarray(sim_no_loss).ravel()

    eps = 1e-12
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((x.size - 1) * std_x**2 + (y.size - 1) * std_y**2) / max(x.size + y.size - 2, 1))
    cohen_d = (mean_x - mean_y) / (pooled + eps)
    cliffs = _cliffs_delta(x, y)

    stats = {
        'n_with': int(x.size),
        'n_without': int(y.size),
        'mean_with': float(mean_x),
        'mean_without': float(mean_y),
        'median_with': float(np.median(x)),
        'median_without': float(np.median(y)),
        'iqr_with': float(np.percentile(x, 75) - np.percentile(x, 25)),
        'iqr_without': float(np.percentile(y, 75) - np.percentile(y, 25)),
        'cohen_d': float(cohen_d),
        'cliffs_delta': cliffs,
    }

    if ks_2samp is not None:
        ks = ks_2samp(x, y, alternative='two-sided')
        stats['ks_d'] = float(ks.statistic)
        stats['ks_p'] = float(ks.pvalue)

    if mannwhitneyu is not None:
        mw = mannwhitneyu(x, y, alternative='two-sided')
        stats['mw_u'] = float(mw.statistic)
        stats['mw_p'] = float(mw.pvalue)

    if wilcoxon is not None:
        min_len = min(x.size, y.size)
        if min_len > 0:
            try:
                w = wilcoxon(x[:min_len], y[:min_len], alternative='two-sided', zero_method='wilcox')
                stats['wilcoxon_w'] = float(w.statistic)
                stats['wilcoxon_p'] = float(w.pvalue)
            except Exception:
                pass
    return stats


def _annotate_stats(ax, stats, y_anchor=0.98):
    lines = [
        f"n={stats['n_with']} vs {stats['n_without']}",
        f"mean={stats['mean_with']:.3f} vs {stats['mean_without']:.3f}",
        f"median={stats['median_with']:.3f} vs {stats['median_without']:.3f}",
        f"Cohen's d={stats['cohen_d']:.3f}",
        f"Cliff's δ={stats['cliffs_delta']:.3f}",
    ]
    if 'ks_d' in stats and 'ks_p' in stats:
        lines.append(f"KS D={stats['ks_d']:.3f}, p={stats['ks_p']:.2e}")
    if 'mw_u' in stats and 'mw_p' in stats:
        lines.append(f"MW-U p={stats['mw_p']:.2e}")

    ax.text(
        0.98,
        y_anchor,
        '\n'.join(lines),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
    )


def _save_figure(fig, out_prefix):
    fig.savefig(f'{out_prefix}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{out_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved -> {out_prefix}.pdf & {out_prefix}.png")


def _plot_kde_hist(sim_with_loss, sim_no_loss, stats, out_prefix):
    configure_academic_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(-0.5, 1.0, 31)
    ax.hist(sim_no_loss, bins=bins, density=True, alpha=0.25, color='#f59622', label='w/o Orthogonality Loss')
    ax.hist(sim_with_loss, bins=bins, density=True, alpha=0.25, color='#1f77b4', label='Full ACML')
    sns.kdeplot(sim_no_loss, color='#f59622', linewidth=2.2, ax=ax)
    sns.kdeplot(sim_with_loss, color='#1f77b4', linewidth=2.2, ax=ax)

    ax.axvline(np.mean(sim_no_loss), color='#f59622', linestyle='--', linewidth=2)
    ax.axvline(np.mean(sim_with_loss), color='#1f77b4', linestyle='--', linewidth=2)

    ax.set_xlabel('Cosine Similarity', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)
    ax.tick_params(axis='both', labelsize=23)
    ax.set_xlim(-0.5, 1.0)

    _annotate_stats(ax, stats)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=1, fontsize=20)
    # 删除标题
    fig.tight_layout()
    _save_figure(fig, out_prefix)


def _plot_ecdf(sim_with_loss, sim_no_loss, stats, out_prefix):
    configure_academic_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.ecdfplot(sim_no_loss, color='#f59622', linewidth=2.6, label='w/o Orthogonality Loss', ax=ax)
    sns.ecdfplot(sim_with_loss, color='#1f77b4', linewidth=2.6, label='Full ACML', ax=ax)

    ax.set_xlabel('Cosine Similarity', fontsize=30)
    ax.set_ylabel('Cumulative Probability', fontsize=30)
    ax.tick_params(axis='both', labelsize=23)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0.0, 1.0)

    _annotate_stats(ax, stats)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1, fontsize=20)
    # 删除标题
    fig.tight_layout()
    _save_figure(fig, out_prefix)


def _plot_violin_box(sim_with_loss, sim_no_loss, stats, out_prefix):
    configure_academic_style()

    labels = (['w/o Orthogonality Loss'] * len(sim_no_loss)) + (['Full ACML'] * len(sim_with_loss))
    values = np.concatenate([sim_no_loss, sim_with_loss])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        x=labels,
        y=values,
        order=['w/o Orthogonality Loss', 'Full ACML'],
        palette=['#f59622', '#1f77b4'],
        inner=None,
        cut=0,
        linewidth=1.5,
        ax=ax,
    )
    sns.boxplot(
        x=labels,
        y=values,
        order=['w/o Orthogonality Loss', 'Full ACML'],
        width=0.22,
        showcaps=True,
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 3},
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black', 'linewidth': 2},
        showfliers=False,
        ax=ax,
    )

    ax.set_xlabel('Setting')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(-0.5, 1.0)
    _annotate_stats(ax, stats, y_anchor=0.96)
    ax.set_title('Version C: Violin + Box (Robust Summary)')
    fig.tight_layout()
    _save_figure(fig, out_prefix)


def _plot_paired_delta(sim_with_loss, sim_no_loss, stats, out_prefix):
    configure_academic_style()
    min_len = min(len(sim_with_loss), len(sim_no_loss))
    if min_len == 0:
        print('✗ Skip paired plot: empty data')
        return

    with_arr = np.asarray(sim_with_loss[:min_len])
    no_arr = np.asarray(sim_no_loss[:min_len])
    delta = with_arr - no_arr

    sample_n = min_len
    sample_idx = np.arange(min_len)

    # -------- Figure 1: Paired Slope Plot --------
    fig1, ax1 = plt.subplots(figsize=(6, 4.8))

    x_pos = np.array([0, 1])
    for idx in sample_idx:
        ax1.plot(x_pos, [no_arr[idx], with_arr[idx]], color='gray', alpha=0.08, linewidth=0.5)
    ax1.scatter(np.zeros(sample_n), no_arr[sample_idx], color='#B22222', s=6, alpha=0.25, label='w/o Orthogonality Loss')
    ax1.scatter(np.ones(sample_n), with_arr[sample_idx], color='#1F4788', s=6, alpha=0.25, label='Full ACML')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['w/o Ortho', 'w/ Ortho'], fontsize=21)
    ax1.set_ylabel('Cosine Similarity', fontsize=24)
    ax1.tick_params(axis='y', labelsize=21)
    ax1.tick_params(axis='x', pad=10)  # 增加x轴标签与坐标轴的距离
    ax1.set_ylim(-0.55, 1.05)  # 稍微扩大y轴范围，避免重叠
    ax1.legend(loc='upper right', frameon=False, fontsize=16)  # 去掉边框
    # 设置深黑色框线
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    # 删除标题

    fig1.tight_layout(pad=1.5)  # 增加整体padding
    _save_figure(fig1, f"{out_prefix}_slope")

    # -------- Figure 2: Delta Distribution --------
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))  # 增加宽度以容纳图例

    # 使用ax.hist直接控制颜色，facecolor明确指定填充颜色
    ax2.hist(delta, bins=30, facecolor='#1F4788', alpha=0.35, edgecolor='#1F4788', linewidth=0.8)
    # 添加KDE曲线
    from scipy import stats
    kde = stats.gaussian_kde(delta)
    x_range = np.linspace(delta.min(), delta.max(), 200)
    kde_values = kde(x_range)
    # 缩放KDE以匹配直方图的高度
    hist_counts, _ = np.histogram(delta, bins=30)
    scale_factor = hist_counts.max() / kde_values.max()
    ax2.plot(x_range, kde_values * scale_factor, color='#1F4788', linewidth=2)
    
    ax2.axvline(0.0, color='black', linestyle='--', linewidth=1.5, label='Zero Reference')
    mean_val = np.mean(delta)
    ax2.axvline(mean_val, color='#1F4788', linestyle='--', linewidth=2, label='Mean $\\Delta$')
    ax2.set_xlabel('Δ Similarity', fontsize=24)
    ax2.set_ylabel('Count', fontsize=24)
    ax2.tick_params(axis='both', labelsize=21)
    ax2.tick_params(axis='x', pad=10)  # 增加x轴标签与坐标轴的距离
    # 设置深黑色框线
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    # 删除标题
    # 图例放在右上角，去掉边框，不使用bbox_to_anchor
    ax2.legend(loc='upper right', frameon=False, fontsize=13)

    fig2.tight_layout(pad=1.5)  # 增加整体padding
    _save_figure(fig2, f"{out_prefix}_dist")


def plot_complementarity_variants(sim_with_loss, sim_no_loss, mode='all'):
    """
    mode:
        - 'paired'     : 版本D，配对变化图
    """
    stats = _compute_stats(sim_with_loss, sim_no_loss)
    print('\n[Stats Summary]')
    print(f"  n(with / without): {stats['n_with']} / {stats['n_without']}")
    print(f"  mean(with / without): {stats['mean_with']:.4f} / {stats['mean_without']:.4f}")
    print(f"  median(with / without): {stats['median_with']:.4f} / {stats['median_without']:.4f}")
    print(f"  Cohen's d: {stats['cohen_d']:.4f}")
    print(f"  Cliff's delta: {stats['cliffs_delta']:.4f}")
    if 'ks_p' in stats:
        print(f"  KS p-value: {stats['ks_p']:.4e}")
    if 'mw_p' in stats:
        print(f"  Mann-Whitney p-value: {stats['mw_p']:.4e}")
    if 'wilcoxon_p' in stats:
        print(f"  Wilcoxon p-value: {stats['wilcoxon_p']:.4e}")

    if mode != 'paired':
        print(f"[Info] Requested mode='{mode}' ignored. This script is now fixed to Version D only.")
    _plot_paired_delta(sim_with_loss, sim_no_loss, stats, out_prefix='fig4_complementarity_vD_paired_delta')


def main(model_path, fold_index=2, mode='all'):
    """主函数"""
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if getattr(args, 'modality', None) != 'both':
        print(f"[Info] Override args.modality from '{args.modality}' to 'both' for complementarity plotting.")
        args.modality = 'both'
    
    print(f"\n[1/4] Loading model from {Path(model_path).name}")
    model = LieDetection(args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ Model loaded")
    
    print(f"\n[2/4] Loading test set (fold {fold_index + 1})...")
    _, test_loader, _, _ = create_dolos_dataloaders(
        feature_root=args.feature_root,
        fold_path=args.fold_path,
        collate_fn=dolos_collate_fn,
        fold_index=fold_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_feature_root=args.audio_feature_root,
        modality=args.modality,
        audio_dim=args.audio_dim,
    )
    print(f"✓ Test set loaded")
    
    print(f"\n[3/4] Collecting similarity distributions...")
    sim_with_loss, sim_no_loss = collect_similarity_from_testset(model, test_loader, device)
    print(f"✓ Collected {len(sim_with_loss)} samples")
    print(f"  Mean similarity w/ loss: {np.mean(sim_with_loss):.4f}")
    print(f"  Mean similarity w/o loss: {np.mean(sim_no_loss):.4f}")
    
    print(f"\n[4/4] Plotting mode: {mode}")
    plot_complementarity_variants(sim_with_loss, sim_no_loss, mode=mode)
    print(f"✓ Complete!")


if __name__ == '__main__':
    # ===== 配置 =====
    MODEL_PATH = "/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/photo/model_best_fold_3_98.pth.tar"
    FOLD_INDEX = 2  # 使用第3折（0-indexed）
    PLOT_MODE = 'paired'  # fixed: only Version D
    # ================

    main(MODEL_PATH, FOLD_INDEX, mode=PLOT_MODE)


