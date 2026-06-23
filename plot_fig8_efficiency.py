import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

os.makedirs('figures', exist_ok=True)

# 顶会美学设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Data extracted from Table 1
# format: Name: [Params(M), FLOPs(M), DOLOS_ACC(%)]
# (Convert G to M: 1G = 1000M)
data = {
    'FFCSN': [36.06, 29960, 67.58],
    'FacialCueNet': [22.23, 25900, 63.93],
    'MAD-Net': [0.27, 69.30, 65.40],
    'LSTM-Concat': [0.15, 14.45, 67.83],
    'DECEPTIcON': [1.64, 1.64, 74.45],
    'Graph-CrossModal': [40.31, 180, 52.79],
    'LoRA-Calib': [0.22, 15.40, 65.71],
    'Stacked-BiLSTM': [10.77, 43250, 70.45],
    'DeiT-Wav2Vec': [0.033, 0.03, 68.54],
    'ACML (Ours)': [3.45, 2.94, 75.55]
}

names = list(data.keys())
params = np.array([data[k][0] for k in names])
flops = np.array([data[k][1] for k in names])
acc = np.array([data[k][2] for k in names])

fig, ax = plt.subplots(figsize=(8, 6))

# 使用较小的点和标签，将不同类别划分颜色
# 我们的模型标红并加星型，其他灰色/蓝色
for i, name in enumerate(names):
    if name == 'ACML (Ours)':
        color = '#c1440e' # 显眼的深红/橘色
        marker = 'p' # 五角星/多边形
        edge = 'black'
        s = 400 + params[i] * 10
        zorder = 10
    elif name == 'DECEPTIcON':
        color = '#1f77b4' # 次好模型给个蓝
        marker = 'o'
        edge = 'black'
        s = 150 + params[i] * 10
        zorder = 5
    else:
        color = '#b0b0b0' # 灰色
        marker = 'o'
        edge = 'white'
        s = 100 + params[i] * 10
        zorder = 3
        
    ax.scatter(flops[i], acc[i], color=color, s=s, marker=marker, 
               edgecolors=edge, linewidths=1.5, alpha=0.85, zorder=zorder)
    
    # 标签防遮挡
    text_y = acc[i] + 1.2
    text_x = flops[i]
    if name == 'ACML (Ours)':
        text_y = acc[i] + 1.0
    elif name == 'DECEPTIcON':
        text_y = acc[i] - 1.8
    elif name == 'DeiT-Wav2Vec':
        text_y = acc[i] - 1.5
    elif name == 'Graph-CrossModal':
        text_y = acc[i] + 1.5
        
    fontweight = 'bold' if name == 'ACML (Ours)' else 'normal'
    ax.text(text_x, text_y, name, fontsize=11, ha='center', va='center', 
            fontweight=fontweight, zorder=11, color='black' if name != 'ACML (Ours)' else color)

# FLOPs跨度太大，必须使用对数坐标
ax.set_xscale('log')
ax.set_xlabel('Computational Cost (FLOPs in Millions)', fontsize=15, fontweight='bold')
ax.set_ylabel('Accuracy on DOLOS Dataset (%)', fontsize=15, fontweight='bold')

# 设置网格和边框
ax.grid(True, which="both", ls="--", alpha=0.3, linewidth=1)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# 画一个隐形的点放在图例说明size
msizes = [10, 40]
for ms in msizes:
    ax.scatter([], [], c='gray', alpha=0.5, s=100 + ms*10, label=f'{ms}M Params', edgecolors='white')
ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Circle Size = Parameters', 
          loc='lower right', title_fontsize=12, fontsize=11)

plt.tight_layout()
save_path = os.path.join('figures', 'fig8_efficiency_vs_acc.pdf')
plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
print(f"✅ Efficiency vs Accuracy plot saved to {save_path}")