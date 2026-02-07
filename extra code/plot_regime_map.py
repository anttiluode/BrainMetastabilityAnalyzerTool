#!/usr/bin/env python3
"""
Φ-DWELL: REGIME MAP PLOTTER
===========================
Generates the "Killer Figure" from Alzheimer's results.

Visualizes the "Slippage" Hypothesis:
  X-Axis: Vocabulary Size (Number of states visited)
  Y-Axis: Stability (Top-5 Concentration or Criticality)

Prediction:
  - Healthy (Blue): Low Vocab, High Stability (The "Dwelling" Brain)
  - AD (Red): High Vocab, Low Stability (The "Slipping" Brain)
"""

import json
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_regime_map.py <results.json>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    
    print(f"Loading results from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
        
    # Extract data points
    # Structure might be dict {subj: data} or list [data]
    if isinstance(data, dict):
        items = data.items()
    else:
        # If it's a list, we need to extract IDs somehow, or just enumerate
        items = enumerate(data)

    groups = {'CN': [], 'AD': [], 'FTD': []}
    
    for subj_id, metrics in items:
        # Determine group from ID (sub-001 is AD, sub-037 is CN, sub-066 is FTD)
        # Or check if 'group' key exists
        group = metrics.get('group')
        if not group:
            # Fallback logic based on dataset numbering if group missing
            try:
                sid = int(subj_id.split('-')[-1].split('_')[0])
                if sid <= 36: group = 'AD'
                elif sid <= 65: group = 'CN'
                else: group = 'FTD'
            except:
                continue
        
        # We want Vocabulary vs Top-5 Concentration
        vocab = metrics.get('vocabulary_size')
        stability = metrics.get('top5_concentration') # Or 'mean_cv'
        
        if vocab and stability:
            groups[group].append((vocab, stability))

    # PLOT
    plt.figure(figsize=(10, 8), facecolor='#0a0a12')
    ax = plt.gca()
    ax.set_facecolor('#0a0a12')
    
    # Define Styles
    styles = {
        'CN': {'color': '#00ccff', 'label': 'Healthy (CN)', 'marker': 'o'},
        'AD': {'color': '#ff3333', 'label': 'Alzheimer\'s (AD)', 'marker': 'x'},
        'FTD': {'color': '#ffcc00', 'label': 'Frontotemporal (FTD)', 'marker': '^'}
    }
    
    print("\nPlotting Data Points:")
    print(f"{'GROUP':<5} {'N':<5} {'MEAN VOCAB':<12} {'MEAN STABILITY'}")
    print("-" * 40)
    
    for g_name, points in groups.items():
        if not points: continue
        
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]
        
        s = styles[g_name]
        plt.scatter(x, y, c=s['color'], label=s['label'], 
                    marker=s['marker'], s=80, alpha=0.8, edgecolors='none')
        
        # Add Ellipse/Centroid
        mean_x, mean_y = np.mean(x), np.mean(y)
        plt.scatter(mean_x, mean_y, c=s['color'], s=300, marker='+', linewidth=3)
        
        print(f"{g_name:<5} {len(points):<5} {mean_x:<12.1f} {mean_y:.3f}")

    # Decorate
    plt.title("THE REGIME MAP: Brain Stability vs. Vocabulary", 
              fontsize=16, color='white', fontweight='bold', pad=20)
    plt.xlabel("Vocabulary Size (Number of Eigenmode Words)", fontsize=12, color='white')
    plt.ylabel("Stability (Top-5 Concentration)", fontsize=12, color='white')
    
    plt.grid(color='white', alpha=0.1)
    plt.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')
    
    plt.legend(facecolor='#0a0a12', labelcolor='white', loc='upper right')
    
    # Annotate Zones
    plt.text(0.1, 0.9, "STABLE ZONE\n(Healthy)", transform=ax.transAxes, 
             color='#00ccff', alpha=0.5, fontsize=14, fontweight='bold')
    
    plt.text(0.8, 0.1, "SLIPPAGE ZONE\n(Pathology)", transform=ax.transAxes, 
             color='#ff3333', alpha=0.5, fontsize=14, fontweight='bold', ha='right')

    out_path = "phidwell_regime_map.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor='#0a0a12')
    print(f"\nSaved Regime Map: {out_path}")

if __name__ == "__main__":
    main()