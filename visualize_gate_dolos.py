"""
DOLOS Dataset - Dynamic Gated Fusion Visualization
Visualizes the gate activation weights (g) under strong vs weak visual evidence.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from models.photo import LieDetection
from configs.dolos import Args
from datasets.dolos import DOLOSDataset, dolos_collate_fn
from torch.utils.data import DataLoader

def extract_gate_and_certainty(model, dataloader, device):
    """
    Extracts gate activation values and visual certainty scores for all samples.
    """
    model.eval()
    
    gate_values = []
    vis_certainties = []
    
    # 1. Register a forward hook to intercept the gate values
    # In models.photo, the gate module is 'audio_gate'
    captured_gates = []
    def gate_hook(module, inp, output):
        # output is the gate matrix of shape [B, D]. We take mean across features to get a scalar gate activation.
        captured_gates.append(output.detach().cpu())
        
    hook_handle = model.audio_gate.register_forward_hook(gate_hook)
    
    with torch.no_grad():
        for visual_list, audio_list, labels, names in tqdm(dataloader, desc="Extracting features"):
            # Move to device
            visual_list = [v.to(device) if v is not None else None for v in visual_list]
            audio_list = [a.to(device) if a is not None else None for a in audio_list]
            
            # Clear captured gates for this batch
            captured_gates.clear()
            
            # Forward pass
            outputs = model(visual_list, audio_list, labels)
            
            # Get the gate values from the hook
            batch_gates = captured_gates[0].mean(dim=-1).numpy() # shape: [B]
            
            # Get visual instance logits to compute visual certainty
            inst_logits_list = outputs['inst_logits'] # list of [N, 2]
            
            for i, logits in enumerate(inst_logits_list):
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1) # [N, 2]
                
                # Visual certainty: the maximum confidence of any instance belonging to any class,
                # or the confidence of the mean pooling. 
                # Here we use the max probability among instances as the evidence strength.
                max_confidence = probs.max().item()
                
                vis_certainties.append(max_confidence)
                gate_values.append(batch_gates[i])
                
    # Remove hook
    hook_handle.remove()
    
    return np.array(gate_values), np.array(vis_certainties)

def plot_gate_distribution(gate_values, vis_certainties, save_dir):
    """
    Plots the violin plot of gate activation weights for strong vs weak visual evidence.
    """
    print("\nPreparing visualization...")
    
    # Split the samples based on visual certainty percentiles (Strong: Top 33%, Weak: Bottom 33%)
    q_high = np.percentile(vis_certainties, 67)
    q_low = np.percentile(vis_certainties, 33)
    
    strong_vis_gates = gate_values[vis_certainties >= q_high]
    weak_vis_gates = gate_values[vis_certainties <= q_low]
    
    print(f"  - Total samples: {len(gate_values)}")
    print(f"  - Strong visual evidence samples: {len(strong_vis_gates)}")
    print(f"  - Weak visual evidence samples: {len(weak_vis_gates)}")
    print(f"  - Mean gate (Strong): {strong_vis_gates.mean():.4f}")
    print(f"  - Mean gate (Weak):   {weak_vis_gates.mean():.4f}")
    
    # Figure setups (Top-tier conference style)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create violin plot
    parts = ax.violinplot([strong_vis_gates, weak_vis_gates], 
                          showmeans=True, showmedians=False, showextrema=True)
    
    # Colors: Professional Palette
    colors = ['#1f4788', '#c1440e']  # Deep Blue (Strong), Brick Red (Weak)
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2.0)
        
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Strong Visual\nEvidence', 'Weak Visual\nEvidence'], 
                       fontsize=18, fontweight='bold')
    
    ax.set_ylabel(r'Gate Activation Weight ($\mathbf{g}$)', 
                  fontsize=18, fontweight='bold')
    
    # Set ylim to fit data [0, 1] usually since it's a sigmoid gate
    ax.set_ylim(-0.05, 1.05)
    
    # Grid and spines
    ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=1)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    save_path = str(save_dir / 'gate_distribution_real.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ Figure saved to: {save_path}")
    plt.close()

def main():
    # Setup Paths & Args
    model_path = "/home/pengxf/work/TDD/Video_MAEV2/Deception/experiments/photo/model_best_fold_3_98.pth.tar"
    fold_index = 2  # Adjust if model_best_fold_3 means fold 2 (0-indexed)
    
    args = Args()
    # Ensure modal is set to 'both' to activate the fusion gate!
    args.modality = 'both'
    args.hidden_dim = 768
    args.low_dim = 128
    
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        print(f"Checkpoint Modality: {getattr(saved_args, 'modality', 'N/A')}")
        args.modality = 'both' # Force 'both' for visualization
        
    model = LieDetection(args).to(device)
    
    # Adjust state dict in case strict throws errors due to mismatches in older saves
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if unexpected:
        print(f"⚠️ Ignored {len(unexpected)} unexpected keys")
    print("✅ Model loaded successfully")
    
    # Load Dataset
    print(f"\nLoading DOLOS fold {fold_index + 1} test dataset...")
    _, test_dataset = DOLOSDataset.create_train_test_datasets(
        feature_root=args.feature_root,
        fold_path=args.fold_path,
        fold_index=fold_index,
        audio_feature_root=args.audio_feature_root,
        modality='both',
        audio_dim=args.audio_dim
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             num_workers=0, collate_fn=dolos_collate_fn)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Extract
    gate_values, vis_certainties = extract_gate_and_certainty(model, test_loader, device)
    
    # Visualize
    plot_gate_distribution(gate_values, vis_certainties, output_dir)
    print("\n🎉 Gate visualization complete!")

if __name__ == "__main__":
    main()
