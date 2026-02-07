#!/usr/bin/env python3
"""
Φ-Dwell Cross-Subject Analyzer
================================
Loads fingerprints from N subjects and answers:

1. UNIVERSALS: What is constant across all brains?
2. INDIVIDUALS: What varies and how much?
3. CLUSTERS: Do brains group into types?
4. SIGNATURES: What makes each brain unique?
5. STATISTICS: Which measures have significant cross-subject variance?

Usage:
    python phidwell_cross_subject.py path/to/phidwell_all_fingerprints.json
    python phidwell_cross_subject.py path/to/dir/with/individual/fingerprints/
"""

import numpy as np
import json
import os
import sys
import glob
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


def load_fingerprints(path):
    """Load fingerprints from either a combined JSON or a directory of individual JSONs."""
    fingerprints = {}
    
    if os.path.isfile(path) and path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        # Could be {subject: {key: val}} or a list
        if isinstance(data, dict):
            # Check if it's a single fingerprint or multi-subject
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                fingerprints = data
            else:
                fingerprints = {'single': data}
    elif os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, 'phidwell_fingerprint_*.json'))):
            name = os.path.basename(fp).replace('phidwell_fingerprint_', '').replace('.json', '')
            with open(fp) as f:
                fingerprints[name] = json.load(f)
    
    return fingerprints


def fingerprints_to_matrix(fingerprints):
    """Convert fingerprints dict to a feature matrix."""
    subjects = sorted(fingerprints.keys())
    # Get all keys from first subject
    all_keys = sorted(fingerprints[subjects[0]].keys())
    
    # Build matrix
    matrix = np.zeros((len(subjects), len(all_keys)))
    for i, subj in enumerate(subjects):
        for j, key in enumerate(all_keys):
            val = fingerprints[subj].get(key, 0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0
            matrix[i, j] = val
    
    return subjects, all_keys, matrix


def analyze_universals(subjects, keys, matrix):
    """Find what's universal vs variable across subjects."""
    n_subj = len(subjects)
    
    results = []
    for j, key in enumerate(keys):
        col = matrix[:, j]
        col_clean = col[~np.isnan(col)]
        if len(col_clean) < 3:
            continue
        
        mean = np.mean(col_clean)
        std = np.std(col_clean)
        cv = std / (abs(mean) + 1e-10)
        min_v = np.min(col_clean)
        max_v = np.max(col_clean)
        
        # One-sample t-test: is the mean significantly different from 0?
        if std > 0:
            t_stat, p_val = stats.ttest_1samp(col_clean, 0)
        else:
            t_stat, p_val = 0, 1
        
        results.append({
            'key': key,
            'mean': mean,
            'std': std,
            'cv': cv,
            'min': min_v,
            'max': max_v,
            'range': max_v - min_v,
            't_stat': t_stat,
            'p_val': p_val,
        })
    
    return results


def find_clusters(subjects, keys, matrix):
    """Cluster subjects by fingerprint similarity."""
    # Remove columns with zero variance
    col_std = np.std(matrix, axis=0)
    valid = col_std > 1e-10
    mat_valid = matrix[:, valid]
    keys_valid = [k for k, v in zip(keys, valid) if v]
    
    # Normalize
    mean = np.mean(mat_valid, axis=0)
    std = np.std(mat_valid, axis=0)
    std[std < 1e-10] = 1
    mat_norm = (mat_valid - mean) / std
    
    # Distance matrix
    if len(subjects) >= 3:
        dists = pdist(mat_norm, metric='euclidean')
        dist_matrix = squareform(dists)
        
        # Hierarchical clustering
        Z = linkage(dists, method='ward')
        
        # Cut at 2 and 3 clusters
        labels_2 = fcluster(Z, t=2, criterion='maxclust')
        labels_3 = fcluster(Z, t=3, criterion='maxclust')
        
        return {
            'dist_matrix': dist_matrix,
            'linkage': Z,
            'labels_2': labels_2,
            'labels_3': labels_3,
            'mat_norm': mat_norm,
            'keys_valid': keys_valid,
        }
    return None


def find_discriminating_features(subjects, keys, matrix, labels):
    """Find features that most differentiate clusters."""
    n_clusters = len(set(labels))
    if n_clusters < 2:
        return []
    
    results = []
    for j, key in enumerate(keys):
        col = matrix[:, j]
        groups = [col[labels == c] for c in sorted(set(labels))]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) >= 2:
            if len(groups) == 2:
                stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                stat, p = stats.kruskal(*groups)
            
            results.append({
                'key': key,
                'stat': stat,
                'p': p,
                'group_means': [float(np.mean(g)) for g in groups],
            })
    
    results.sort(key=lambda x: x['p'])
    return results


def compute_subject_signatures(subjects, keys, matrix):
    """For each subject, find their most distinctive features."""
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std < 1e-10] = 1
    z_scores = (matrix - mean) / std
    
    signatures = {}
    for i, subj in enumerate(subjects):
        # Top 5 most extreme z-scores
        z = z_scores[i]
        extreme_idx = np.argsort(np.abs(z))[::-1][:5]
        sig = [(keys[j], float(z[j]), float(matrix[i, j])) for j in extreme_idx]
        signatures[subj] = sig
    
    return signatures, z_scores


def create_cross_subject_figure(subjects, keys, matrix, universals, 
                                  clusters, signatures, z_scores, output_path):
    """Generate the comprehensive cross-subject analysis figure."""
    
    fig = plt.figure(figsize=(28, 22), facecolor='#0a0a12')
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.4,
                  left=0.05, right=0.97, top=0.93, bottom=0.04)
    
    n_subj = len(subjects)
    short_names = [s.replace('R01', '') for s in subjects]
    
    fig.suptitle(
        f'Φ-DWELL CROSS-SUBJECT ANALYSIS   ·   {n_subj} subjects   ·   '
        f'{len(keys)} fingerprint dimensions',
        fontsize=14, fontweight='bold', color='#b44adc',
        fontfamily='monospace', y=0.97
    )
    
    # ══════════════════════════════════════════════════
    # Panel 1: UNIVERSALS - Band hierarchy (self-transition rates)
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_labels = ['δ', 'θ', 'α', 'β', 'γ']
    band_colors = ['#ff5050', '#ffb428', '#50ff50', '#50a0ff', '#c850ff']
    
    self_trans_keys = [f'{b}_self_trans_mean' for b in band_names]
    for bi, (key, label, color) in enumerate(zip(self_trans_keys, band_labels, band_colors)):
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            vals = matrix[:, j]
            ax.boxplot([vals], positions=[bi], widths=0.6,
                      boxprops=dict(color=color), medianprops=dict(color='white'),
                      whiskerprops=dict(color=color), capprops=dict(color=color),
                      flierprops=dict(markeredgecolor=color, markersize=3))
            ax.scatter(np.ones(n_subj) * bi + np.random.randn(n_subj) * 0.05,
                      vals, c=color, s=15, alpha=0.6, zorder=5)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(band_labels, fontsize=11, color='#aaaacc')
    ax.set_ylabel('Self-transition rate', fontsize=8, color='#5a5a7a')
    ax.set_title('UNIVERSAL: Band Persistence Hierarchy', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=7)
    
    # ══════════════════════════════════════════════════
    # Panel 2: UNIVERSALS - Cross-band coupling hierarchy
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])
    
    coupling_keys = []
    coupling_labels = []
    for bi in range(5):
        for bj in range(bi + 1, 5):
            key = f'coupling_{band_names[bi]}_{band_names[bj]}'
            coupling_keys.append(key)
            coupling_labels.append(f'{band_labels[bi]}-{band_labels[bj]}')
    
    coupling_means = []
    coupling_stds = []
    for key in coupling_keys:
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            vals = matrix[:, j]
            vals = vals[~np.isnan(vals)]
            coupling_means.append(np.mean(vals) if len(vals) > 0 else 0)
            coupling_stds.append(np.std(vals) if len(vals) > 0 else 0)
        else:
            coupling_means.append(0)
            coupling_stds.append(0)
    
    x = range(len(coupling_keys))
    sort_idx = np.argsort(coupling_means)[::-1]
    bars = ax.bar(range(len(sort_idx)), [coupling_means[i] for i in sort_idx],
                  yerr=[coupling_stds[i] for i in sort_idx],
                  color='#b44adc', alpha=0.7, capsize=3, error_kw=dict(color='#5a5a7a'))
    ax.set_xticks(range(len(sort_idx)))
    ax.set_xticklabels([coupling_labels[i] for i in sort_idx], fontsize=7, 
                        color='#aaaacc', rotation=45)
    ax.set_ylabel('Mean |coupling|', fontsize=8, color='#5a5a7a')
    ax.set_title('UNIVERSAL: Cross-Band Coupling', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=7)
    
    # ══════════════════════════════════════════════════
    # Panel 3: UNIVERSALS - Key metrics distribution
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 2])
    
    key_metrics = ['global_metastability', 'attractor_entropy', 'fraction_critical',
                   'top_attractor_fraction', 'n_attractors_used']
    metric_labels = ['Metastability', 'Att. Entropy', 'Frac Critical', 
                     'Top Attr Frac', '# Attractors']
    
    for mi, (key, label) in enumerate(zip(key_metrics, metric_labels)):
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            vals = matrix[:, j]
            ax.boxplot([vals], positions=[mi], widths=0.6, vert=True,
                      boxprops=dict(color='#50ffb4'), medianprops=dict(color='white'),
                      whiskerprops=dict(color='#50ffb4'), capprops=dict(color='#50ffb4'),
                      flierprops=dict(markeredgecolor='#50ffb4', markersize=3))
    
    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(metric_labels, fontsize=6, color='#aaaacc', rotation=30)
    ax.set_title('UNIVERSAL: Key Metrics (all subjects)', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=7)
    
    # ══════════════════════════════════════════════════
    # Panel 4: Coefficient of Variation - what varies most?
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 3])
    
    # Sort universals by CV
    uni_sorted = sorted(universals, key=lambda x: x['cv'], reverse=True)
    top_variable = uni_sorted[:15]
    names = [u['key'].replace('coupling_', 'c:').replace('_self_trans', '_st')
             .replace('_n_words', '_nw') for u in top_variable]
    cvs = [u['cv'] for u in top_variable]
    
    ax.barh(range(len(top_variable)), cvs, color='#ff7744', alpha=0.7)
    ax.set_yticks(range(len(top_variable)))
    ax.set_yticklabels(names, fontsize=6, color='#aaaacc', fontfamily='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('CV (higher = more individual)', fontsize=7, color='#5a5a7a')
    ax.set_title('INDIVIDUAL: Most Variable Features', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ══════════════════════════════════════════════════
    # Panel 5: Subject distance matrix
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 0])
    if clusters is not None:
        dm = clusters['dist_matrix']
        im = ax.imshow(dm, cmap='inferno', aspect='auto')
        ax.set_xticks(range(n_subj))
        ax.set_yticks(range(n_subj))
        ax.set_xticklabels(short_names, fontsize=5, color='#aaaacc', rotation=90)
        ax.set_yticklabels(short_names, fontsize=5, color='#aaaacc')
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('DISTANCE MATRIX (fingerprint space)', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    
    # ══════════════════════════════════════════════════
    # Panel 6: Dendrogram
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 1])
    if clusters is not None:
        dendrogram(clusters['linkage'], labels=short_names, ax=ax,
                  leaf_rotation=90, leaf_font_size=6,
                  color_threshold=0, above_threshold_color='#b44adc')
        ax.tick_params(axis='x', colors='#aaaacc', labelsize=6)
        ax.tick_params(axis='y', colors='#3a3a5a', labelsize=6)
    ax.set_title('HIERARCHICAL CLUSTERING', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    
    # ══════════════════════════════════════════════════
    # Panel 7: Subject fingerprint heatmap (z-scores)
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 2:4])
    
    # Select most interesting features
    interesting_keys = []
    for prefix in ['delta_self_trans_mean', 'theta_self_trans_mean', 
                    'alpha_self_trans_mean', 'beta_self_trans_mean', 'gamma_self_trans_mean',
                    'coupling_delta_theta', 'coupling_delta_alpha', 'coupling_theta_alpha',
                    'global_metastability', 'attractor_entropy', 'fraction_critical',
                    'top_attractor_fraction', 'n_attractors_used']:
        if prefix in keys:
            interesting_keys.append(prefix)
    
    idx = [keys.index(k) for k in interesting_keys]
    z_sub = z_scores[:, idx]
    
    im = ax.imshow(z_sub.T, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    ax.set_xticks(range(n_subj))
    ax.set_xticklabels(short_names, fontsize=5, color='#aaaacc', rotation=90)
    ax.set_yticks(range(len(interesting_keys)))
    short_ik = [k.replace('coupling_', 'c:').replace('_self_trans_mean', '_st')
                .replace('_attractor', '_att') for k in interesting_keys]
    ax.set_yticklabels(short_ik, fontsize=6, color='#aaaacc', fontfamily='monospace')
    plt.colorbar(im, ax=ax, fraction=0.03, label='z-score')
    ax.set_title('FINGERPRINT HEATMAP (z-scores, blue=low red=high)', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    
    # ══════════════════════════════════════════════════
    # Panel 8: Global metastability per subject
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 0:2])
    
    j_meta = keys.index('global_metastability') if 'global_metastability' in keys else -1
    j_crit = keys.index('fraction_critical') if 'fraction_critical' in keys else -1
    
    if j_meta >= 0:
        vals = matrix[:, j_meta]
        sort_idx = np.argsort(vals)[::-1]
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_subj))
        
        ax.bar(range(n_subj), vals[sort_idx], color=colors, alpha=0.8)
        ax.set_xticks(range(n_subj))
        ax.set_xticklabels([short_names[i] for i in sort_idx], fontsize=6, 
                            color='#aaaacc', rotation=45)
        
        # Add criticality as line
        if j_crit >= 0:
            ax2 = ax.twinx()
            ax2.plot(range(n_subj), matrix[sort_idx, j_crit], 'o-', 
                    color='#50ffb4', markersize=4, linewidth=1, alpha=0.7)
            ax2.set_ylabel('Fraction critical', fontsize=7, color='#50ffb4')
            ax2.tick_params(colors='#50ffb4', labelsize=6)
        
        ax.set_ylabel('Global metastability (CV)', fontsize=8, color='#5a5a7a')
        mean_meta = np.mean(vals)
        ax.axhline(mean_meta, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.text(n_subj - 1, mean_meta, f' μ={mean_meta:.2f}', fontsize=7, 
                color='white', va='bottom')
    
    ax.set_title('INDIVIDUAL: Global Metastability (+ fraction critical)', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ══════════════════════════════════════════════════
    # Panel 9: Band persistence profiles per subject
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 2:4])
    
    for i in range(n_subj):
        profile = []
        for band in band_names:
            key = f'{band}_self_trans_mean'
            j = keys.index(key) if key in keys else -1
            if j >= 0:
                profile.append(matrix[i, j])
        if len(profile) == 5:
            ax.plot(range(5), profile, 'o-', alpha=0.4, linewidth=1, markersize=3,
                   label=short_names[i] if i < 5 else None)
    
    # Mean profile
    mean_profile = []
    for band in band_names:
        key = f'{band}_self_trans_mean'
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            mean_profile.append(np.mean(matrix[:, j]))
    
    ax.plot(range(5), mean_profile, 'o-', color='white', linewidth=3, 
           markersize=8, zorder=10, label='MEAN')
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(band_labels, fontsize=11, color='#aaaacc')
    ax.set_ylabel('Self-transition rate', fontsize=8, color='#5a5a7a')
    ax.set_title('ALL SUBJECTS: Band Persistence Profile', fontsize=9, 
                  color='#8888aa', fontfamily='monospace')
    ax.legend(fontsize=5, ncol=5, framealpha=0.3, loc='lower left')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=7)
    
    # ══════════════════════════════════════════════════
    # Panel 10: Subject signatures
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, 0:2])
    ax.axis('off')
    ax.set_facecolor('#0f0f1a')
    
    ax.text(0.01, 0.97, 'SUBJECT SIGNATURES (most distinctive features)', 
            fontsize=9, color='#b44adc', fontfamily='monospace', fontweight='bold',
            transform=ax.transAxes, va='top')
    
    y = 0.90
    for i, subj in enumerate(subjects[:10]):
        sig = signatures[subj]
        parts = []
        for key, z, val in sig[:3]:
            short_k = key.replace('coupling_', 'c:').replace('_self_trans', '_st')
            direction = '↑' if z > 0 else '↓'
            parts.append(f'{short_k}{direction}({z:+.1f}σ)')
        line = f'{short_names[i]:>8}: {" | ".join(parts)}'
        color = '#ff7744' if any(abs(z) > 2 for _, z, _ in sig[:3]) else '#8888aa'
        ax.text(0.01, y, line, fontsize=6, color=color, fontfamily='monospace',
                transform=ax.transAxes)
        y -= 0.09
    
    # ══════════════════════════════════════════════════
    # Panel 11: Summary statistics table
    # ══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, 2:4])
    ax.axis('off')
    ax.set_facecolor('#0f0f1a')
    
    ax.text(0.01, 0.97, 'CROSS-SUBJECT SUMMARY', 
            fontsize=9, color='#b44adc', fontfamily='monospace', fontweight='bold',
            transform=ax.transAxes, va='top')
    
    # Compute summary stats
    j_meta = keys.index('global_metastability') if 'global_metastability' in keys else -1
    j_ent = keys.index('attractor_entropy') if 'attractor_entropy' in keys else -1
    j_crit = keys.index('fraction_critical') if 'fraction_critical' in keys else -1
    j_natt = keys.index('n_attractors_used') if 'n_attractors_used' in keys else -1
    
    lines = []
    lines.append(f'N subjects: {n_subj}')
    lines.append(f'N fingerprint dimensions: {len(keys)}')
    lines.append('')
    
    if j_meta >= 0:
        v = matrix[:, j_meta]
        lines.append(f'Global metastability:  {np.mean(v):.3f} ± {np.std(v):.3f}  '
                     f'(range {np.min(v):.3f}-{np.max(v):.3f})')
    if j_ent >= 0:
        v = matrix[:, j_ent]
        lines.append(f'Attractor entropy:     {np.mean(v):.3f} ± {np.std(v):.3f}')
    if j_crit >= 0:
        v = matrix[:, j_crit]
        lines.append(f'Fraction critical:     {np.mean(v):.3f} ± {np.std(v):.3f}')
    if j_natt >= 0:
        v = matrix[:, j_natt]
        lines.append(f'Attractors used:       {np.mean(v):.1f} ± {np.std(v):.1f}')
    
    lines.append('')
    lines.append('BAND PERSISTENCE HIERARCHY (mean ± std):')
    for band, label in zip(band_names, band_labels):
        key = f'{band}_self_trans_mean'
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            v = matrix[:, j]
            lines.append(f'  {label} ({band:>5}): {np.mean(v):.4f} ± {np.std(v):.4f}')
    
    lines.append('')
    lines.append('COUPLING HIERARCHY (mean ± std):')
    coupling_data = []
    for bi in range(5):
        for bj in range(bi + 1, 5):
            key = f'coupling_{band_names[bi]}_{band_names[bj]}'
            j = keys.index(key) if key in keys else -1
            if j >= 0:
                v = matrix[:, j]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    coupling_data.append((f'{band_labels[bi]}-{band_labels[bj]}', 
                                         np.mean(v), np.std(v)))
    coupling_data.sort(key=lambda x: x[1], reverse=True)
    for label, mean, std in coupling_data:
        lines.append(f'  {label}: {mean:.4f} ± {std:.4f}')
    
    # Inter-subject distance stats
    if clusters is not None:
        dm = clusters['dist_matrix']
        upper = dm[np.triu_indices_from(dm, k=1)]
        lines.append('')
        lines.append(f'Inter-subject distance: {np.mean(upper):.2f} ± {np.std(upper):.2f}')
        lines.append(f'  Most similar pair: {np.min(upper):.2f}')
        lines.append(f'  Most different:    {np.max(upper):.2f}')
    
    y = 0.88
    for line in lines:
        ax.text(0.01, y, line, fontsize=6, color='#c8c8e8', fontfamily='monospace',
                transform=ax.transAxes)
        y -= 0.038
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a12')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python phidwell_cross_subject.py <fingerprints.json or dir>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    print("╔══════════════════════════════════════════════════╗")
    print("║   Φ-DWELL CROSS-SUBJECT ANALYZER                ║")
    print("║   Universals · Individuals · Clusters            ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Load
    print(f"\nLoading fingerprints from: {path}")
    fingerprints = load_fingerprints(path)
    print(f"  Found {len(fingerprints)} subjects")
    
    if len(fingerprints) < 2:
        print("Need at least 2 subjects for cross-subject analysis")
        sys.exit(1)
    
    # Convert to matrix
    subjects, keys, matrix = fingerprints_to_matrix(fingerprints)
    print(f"  Feature matrix: {matrix.shape[0]} subjects × {matrix.shape[1]} features")
    
    # Analysis
    print("\n═══ UNIVERSALS ═══")
    universals = analyze_universals(subjects, keys, matrix)
    
    # Most consistent (lowest CV)
    uni_by_cv = sorted(universals, key=lambda x: x['cv'])
    print("\n  Most UNIVERSAL features (lowest CV):")
    for u in uni_by_cv[:10]:
        print(f"    {u['key']:>35}: {u['mean']:.4f} ± {u['std']:.4f} (CV={u['cv']:.3f})")
    
    print("\n  Most INDIVIDUAL features (highest CV):")
    for u in uni_by_cv[-10:][::-1]:
        print(f"    {u['key']:>35}: {u['mean']:.4f} ± {u['std']:.4f} (CV={u['cv']:.3f})")
    
    # Clusters
    print("\n═══ CLUSTERS ═══")
    clusters = find_clusters(subjects, keys, matrix)
    if clusters is not None:
        print(f"  2-cluster labels: {dict(zip([s.replace('R01','') for s in subjects], clusters['labels_2']))}")
        
        disc = find_discriminating_features(subjects, keys, matrix, clusters['labels_2'])
        if disc:
            print(f"\n  Top discriminating features (2 clusters):")
            for d in disc[:5]:
                print(f"    {d['key']:>35}: p={d['p']:.4f}, "
                      f"group means={[f'{m:.3f}' for m in d['group_means']]}")
    
    # Signatures
    print("\n═══ SUBJECT SIGNATURES ═══")
    signatures, z_scores = compute_subject_signatures(subjects, keys, matrix)
    for subj in subjects:
        sig = signatures[subj]
        parts = [f"{k}={v:.3f}(z={z:+.1f})" for k, z, v in sig[:3]]
        print(f"  {subj.replace('R01',''):>8}: {', '.join(parts)}")
    
    # Generate figure
    print("\n═══ GENERATING FIGURE ═══")
    output_dir = os.path.dirname(os.path.abspath(path))
    output_path = os.path.join(output_dir, 'phidwell_cross_subject_analysis.png')
    create_cross_subject_figure(subjects, keys, matrix, universals,
                                  clusters, signatures, z_scores, output_path)
    
    # Print the key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Band hierarchy
    print("\n1. FREQUENCY HIERARCHY (universal across all subjects):")
    for band, label in zip(['delta', 'theta', 'alpha', 'beta', 'gamma'], 
                           ['δ', 'θ', 'α', 'β', 'γ']):
        key = f'{band}_self_trans_mean'
        j = keys.index(key) if key in keys else -1
        if j >= 0:
            v = matrix[:, j]
            print(f"   {label} persistence: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    # Criticality
    j_crit = keys.index('fraction_critical') if 'fraction_critical' in keys else -1
    if j_crit >= 0:
        v = matrix[:, j_crit]
        print(f"\n2. CRITICALITY: {np.mean(v)*100:.0f}% ± {np.std(v)*100:.0f}% of attractors "
              f"are in critical regime (CV > 1.0)")
        print(f"   Range: {np.min(v)*100:.0f}% - {np.max(v)*100:.0f}%")
    
    # Coupling
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_labels = ['δ', 'θ', 'α', 'β', 'γ']
    coupling_data = []
    for bi in range(5):
        for bj in range(bi + 1, 5):
            key = f'coupling_{band_names[bi]}_{band_names[bj]}'
            j = keys.index(key) if key in keys else -1
            if j >= 0:
                v = matrix[:, j]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    coupling_data.append((f'{band_labels[bi]}-{band_labels[bj]}',
                                         np.mean(v), np.std(v)))
    coupling_data.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n3. COUPLING HIERARCHY (δ-θ strongest, β-γ weakest):")
    for label, mean, std in coupling_data[:3]:
        print(f"   {label}: {mean:.4f} ± {std:.4f}")
    print(f"   ...")
    for label, mean, std in coupling_data[-2:]:
        print(f"   {label}: {mean:.4f} ± {std:.4f}")
    
    # Metastability
    j_meta = keys.index('global_metastability') if 'global_metastability' in keys else -1
    if j_meta >= 0:
        v = matrix[:, j_meta]
        print(f"\n4. METASTABILITY: {np.mean(v):.3f} ± {np.std(v):.3f}")
        most = subjects[np.argmax(v)].replace('R01', '')
        least = subjects[np.argmin(v)].replace('R01', '')
        print(f"   Most metastable: {most} ({np.max(v):.3f})")
        print(f"   Least metastable: {least} ({np.min(v):.3f})")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
