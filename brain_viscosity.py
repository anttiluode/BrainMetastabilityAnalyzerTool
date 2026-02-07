#!/usr/bin/env python3
"""
β-Sieve × Φ-Dwell: Brain Viscosity Analyzer
=============================================

Applies the β-gradient "viscosity" concept from grokking research to brain EEG
eigenmode dynamics. The key insight: in neural networks, roughness increases 
across the layer hierarchy when genuine structure is forming (grokking), but 
stays flat during mere memorization.

In the brain, the frequency band hierarchy (δ→θ→α→β→γ) is a natural "depth"
axis — slow oscillations modulate fast ones. We compute:

1. Band-Roughness Profile: How "textured" are eigenmode dynamics at each 
   frequency band? (analog of per-layer roughness)
   
2. β-Gradient (Brain): The slope of roughness across bands — does the brain
   build structured high-frequency texture on top of slow foundations?
   
3. Viscosity Index: Combined measure of how "crystallized" the eigenmode 
   dynamics are — high viscosity = organized, constrained flow (healthy);
   low viscosity = diffuse, random drift (pathological).

Can run in two modes:
  A) From existing phidwell_alzheimer_results.json (fast, reanalyzes saved metrics)
  B) Directly on EEG .set files from ds004504 (full pipeline)

Usage:
  python brain_viscosity.py phidwell_alzheimer_results.json
  python brain_viscosity.py /path/to/ds004504/  [--full]
"""

import sys
import os
import json
import numpy as np
from collections import Counter
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. BRAIN β-GRADIENT: Band Hierarchy as "Depth"
# ============================================================================

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGES = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 45)
}

# Standard 10-20 electrode positions (2D projection)
ELECTRODES_1020 = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'F7': (-0.81, 0.59), 'F3': (-0.39, 0.59), 'Fz': (0.0, 0.59),
    'F4': (0.39, 0.59), 'F8': (0.81, 0.59),
    'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.5, 0.0), 'T4': (1.0, 0.0),
    'T5': (-0.81, -0.59), 'P3': (-0.39, -0.59), 'Pz': (0.0, -0.59),
    'P4': (0.39, -0.59), 'T6': (0.81, -0.59),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
}

CHAN_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2', 'EEG F7': 'F7',
    'EEG F3': 'F3', 'EEG Fz': 'Fz', 'EEG F4': 'F4', 'EEG F8': 'F8',
    'EEG T3': 'T3', 'EEG C3': 'C3', 'EEG Cz': 'Cz', 'EEG C4': 'C4',
    'EEG T4': 'T4', 'EEG T5': 'T5', 'EEG P3': 'P3', 'EEG Pz': 'Pz',
    'EEG P4': 'P4', 'EEG T6': 'T6', 'EEG O1': 'O1', 'EEG O2': 'O2',
}


def build_eigenmodes(n_modes=6, sigma=0.5):
    """Build graph Laplacian eigenmodes from electrode geometry."""
    names = list(ELECTRODES_1020.keys())
    pos = np.array([ELECTRODES_1020[n] for n in names])
    n = len(names)
    
    # Gaussian kernel adjacency
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(np.sum((pos[i] - pos[j])**2))
    W = np.exp(-dist**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)
    
    # Graph Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Skip trivial mode 0, take next n_modes
    modes = eigenvectors[:, 1:n_modes+1]
    return names, modes, eigenvalues[1:n_modes+1]


def map_channels(raw_names, electrode_names):
    """Map raw channel names to standard electrode positions."""
    mapping = {}
    for i, rn in enumerate(raw_names):
        clean = rn.strip()
        if clean in electrode_names:
            mapping[clean] = i
        elif clean in CHAN_ALIASES and CHAN_ALIASES[clean] in electrode_names:
            mapping[CHAN_ALIASES[clean]] = i
        else:
            for alias, standard in CHAN_ALIASES.items():
                if alias.lower() == clean.lower() and standard in electrode_names:
                    mapping[standard] = i
                    break
    return mapping


# ============================================================================
# 2. FULL EEG PIPELINE: β-Gradient on Raw Phase Data
# ============================================================================

def compute_brain_viscosity_from_eeg(data, sfreq, ch_names, electrode_names, 
                                      eigenmodes, max_duration=120, word_step_ms=25):
    """
    Full pipeline: EEG → eigenmode phase projections → band-wise roughness → β-gradient.
    
    This is the brain analog of the grokking β-probe:
    - "Layers" = frequency bands (δ→θ→α→β→γ)  
    - "Activations" = eigenmode projection magnitudes over time
    - "Roughness" = temporal np.diff of eigenmode projections (how jerky/textured)
    - "β-gradient" = slope of roughness across bands (does high-freq build texture?)
    """
    n_modes = eigenmodes.shape[1]
    ch_map = map_channels(ch_names, electrode_names)
    
    if len(ch_map) < 15:
        return None
    
    # Limit duration
    max_samp = int(max_duration * sfreq)
    data = data[:, :max_samp]
    
    # Reorder data to match electrode order
    ordered_data = np.zeros((len(electrode_names), data.shape[1]))
    electrode_list = list(ELECTRODES_1020.keys())
    for i, ename in enumerate(electrode_list):
        if ename in ch_map:
            ordered_data[i] = data[ch_map[ename]]
    
    word_step = max(1, int(word_step_ms * sfreq / 1000))
    n_words = data.shape[1] // word_step
    
    # Per-band eigenmode projections over time
    band_mode_timeseries = {}  # band -> (n_words, n_modes)
    band_dominant_modes = {}   # band -> (n_words,) 
    
    for band_name, (flo, fhi) in BAND_RANGES.items():
        # Bandpass filter
        nyq = sfreq / 2
        if fhi >= nyq:
            fhi = nyq - 1
        try:
            sos = signal.butter(3, [flo/nyq, fhi/nyq], btype='band', output='sos')
            filtered = signal.sosfiltfilt(sos, ordered_data, axis=1)
        except Exception:
            continue
        
        # Hilbert transform → phase
        analytic = signal.hilbert(filtered, axis=1)
        phase = np.angle(analytic)  # (n_channels, n_samples)
        
        # Project onto eigenmodes at each word step
        projections = np.zeros((n_words, n_modes))
        dominants = np.zeros(n_words, dtype=int)
        
        for t in range(n_words):
            sample_idx = t * word_step
            if sample_idx >= phase.shape[1]:
                break
            phase_vec = np.exp(1j * phase[:, sample_idx])
            for m in range(n_modes):
                projections[t, m] = np.abs(np.dot(phase_vec, eigenmodes[:, m]))
            dominants[t] = np.argmax(projections[t])
        
        band_mode_timeseries[band_name] = projections
        band_dominant_modes[band_name] = dominants
    
    if len(band_mode_timeseries) < 5:
        return None
    
    # ================================================================
    # COMPUTE β-GRADIENT METRICS
    # ================================================================
    
    results = {}
    
    # --- A) Per-band roughness (analog of per-layer roughness) ---
    band_roughness = {}
    for band in BANDS:
        if band not in band_mode_timeseries:
            continue
        proj = band_mode_timeseries[band]  # (n_words, n_modes)
        # Normalize (like the NN probe normalizes activations)
        proj_norm = (proj - proj.mean()) / (proj.std() + 1e-6)
        # Temporal roughness: np.diff along time axis
        temporal_diffs = np.diff(proj_norm, axis=0)
        roughness = np.abs(temporal_diffs).mean()
        band_roughness[band] = float(roughness)
    
    results['band_roughness'] = band_roughness
    
    # --- B) β-Gradient: slope of roughness across band hierarchy ---
    # This is the KEY metric. In grokking: deep-shallow roughness difference.
    # In brain: high-freq vs low-freq roughness gradient.
    roughness_values = [band_roughness.get(b, 0) for b in BANDS]
    if len(roughness_values) >= 2:
        # Linear fit across band indices (0=delta through 4=gamma)
        x = np.arange(len(roughness_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, roughness_values)
        results['beta_gradient'] = float(slope)
        results['beta_gradient_r2'] = float(r_value**2)
        results['roughness_slope_p'] = float(p_value)
        # Also compute the simple deep-shallow difference (gamma - delta)
        results['roughness_hilo_diff'] = float(roughness_values[-1] - roughness_values[0])
    
    # --- C) Cross-band roughness coherence ---
    # How correlated are roughness fluctuations across bands?
    # High coherence = bands move together (organized); low = independent (diffuse)
    if len(band_mode_timeseries) >= 2:
        roughness_traces = []
        for band in BANDS:
            if band not in band_mode_timeseries:
                continue
            proj = band_mode_timeseries[band]
            proj_norm = (proj - proj.mean()) / (proj.std() + 1e-6)
            temp_rough = np.abs(np.diff(proj_norm, axis=0)).mean(axis=1)  # per-timestep roughness
            roughness_traces.append(temp_rough)
        
        if len(roughness_traces) >= 2:
            # Pairwise correlation of roughness traces
            min_len = min(len(t) for t in roughness_traces)
            roughness_traces = [t[:min_len] for t in roughness_traces]
            corr_matrix = np.corrcoef(roughness_traces)
            # Mean off-diagonal = cross-band roughness coherence
            n = corr_matrix.shape[0]
            off_diag = corr_matrix[np.triu_indices(n, k=1)]
            results['roughness_coherence'] = float(np.mean(off_diag))
    
    # --- D) Mode-space roughness (spatial texture) ---
    # Instead of temporal diffs, take diffs across eigenmodes at each timestep
    # This measures "spatial roughness" — how textured the eigenmode profile is
    band_spatial_roughness = {}
    for band in BANDS:
        if band not in band_mode_timeseries:
            continue
        proj = band_mode_timeseries[band]
        proj_norm = (proj - proj.mean()) / (proj.std() + 1e-6)
        spatial_diffs = np.diff(proj_norm, axis=1)  # diff across modes
        spatial_roughness = np.abs(spatial_diffs).mean()
        band_spatial_roughness[band] = float(spatial_roughness)
    
    results['band_spatial_roughness'] = band_spatial_roughness
    
    # Spatial β-gradient
    spatial_values = [band_spatial_roughness.get(b, 0) for b in BANDS]
    if len(spatial_values) >= 2:
        x = np.arange(len(spatial_values))
        slope, _, r2, p, _ = stats.linregress(x, spatial_values)
        results['spatial_beta_gradient'] = float(slope)
    
    # --- E) Viscosity Index: combined measure ---
    # High viscosity = organized, constrained, critical
    # Low viscosity = diffuse, random, subcritical
    # Combine: temporal β-gradient × roughness coherence × (1/spatial_gradient)
    bg = results.get('beta_gradient', 0)
    rc = results.get('roughness_coherence', 0)
    sbg = results.get('spatial_beta_gradient', 0.001)
    results['viscosity_index'] = float(abs(bg) * max(rc, 0) / (abs(sbg) + 0.001))
    
    # --- F) Standard Φ-Dwell metrics (recomputed for consistency) ---
    # Tokenize: dominant mode per band → 5-tuple word
    words = []
    min_len_words = min(len(band_dominant_modes[b]) for b in BANDS if b in band_dominant_modes)
    for t in range(min_len_words):
        word = ''.join(str(band_dominant_modes[b][t]+1) for b in BANDS if b in band_dominant_modes)
        if len(word) == 5:
            words.append(word)
    
    if len(words) > 100:
        counts = Counter(words)
        vocab = len(counts)
        total = len(words)
        
        # Entropy
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        
        # Zipf
        sorted_counts = sorted(counts.values(), reverse=True)
        ranks = np.arange(1, len(sorted_counts) + 1)
        log_ranks = np.log(ranks)
        log_counts = np.log(np.array(sorted_counts))
        if len(log_ranks) > 2:
            zipf_slope, _, zipf_r2, _, _ = stats.linregress(log_ranks, log_counts)
            results['zipf_alpha'] = float(-zipf_slope)
        
        # Top-5 concentration
        top5 = sum(sorted_counts[:5]) / total
        
        # Self-transition rate
        self_trans = sum(1 for i in range(1, len(words)) if words[i] == words[i-1]) / (len(words)-1)
        
        # Bigram perplexity
        bigrams = Counter()
        unigrams = Counter()
        for i in range(1, len(words)):
            bigrams[(words[i-1], words[i])] += 1
            unigrams[words[i-1]] += 1
        
        log_pp = 0
        n_pred = 0
        k = 0.01
        for i in range(1, len(words)):
            prev, curr = words[i-1], words[i]
            p_bigram = (bigrams.get((prev, curr), 0) + k) / (unigrams.get(prev, 0) + k * vocab)
            log_pp -= np.log2(p_bigram + 1e-12)
            n_pred += 1
        perplexity = 2 ** (log_pp / max(n_pred, 1))
        
        # Per-band CV (criticality)
        band_cvs = {}
        for band in BANDS:
            if band not in band_dominant_modes:
                continue
            dom = band_dominant_modes[band]
            # Compute dwell times (consecutive runs of same mode)
            dwells = []
            current_run = 1
            for i in range(1, len(dom)):
                if dom[i] == dom[i-1]:
                    current_run += 1
                else:
                    dwells.append(current_run)
                    current_run = 1
            dwells.append(current_run)
            if len(dwells) > 5:
                band_cvs[band] = float(np.std(dwells) / (np.mean(dwells) + 1e-6))
        
        results['vocab_size'] = vocab
        results['entropy'] = float(entropy)
        results['perplexity'] = float(perplexity)
        results['self_rate'] = float(self_trans)
        results['top5_concentration'] = float(top5)
        results['mean_cv'] = float(np.mean(list(band_cvs.values()))) if band_cvs else 0
        results['band_cv'] = band_cvs
        results['n_words'] = len(words)
    
    return results


# ============================================================================
# 3. FROM JSON: Compute β-gradient from pre-existing per-band metrics
# ============================================================================

def compute_beta_from_json(subject_data):
    """
    Compute β-gradient analog from already-computed Φ-Dwell per-band metrics.
    Uses band_cv and band_self_rates as proxies for "roughness" at each band depth.
    
    This is a coarser approximation than the full EEG pipeline but works without
    re-processing raw data.
    """
    results = {}
    
    # Band CV as roughness proxy: 
    # CV measures dwell variability. High CV = critical/textured. Low = regular/flat.
    band_cv = subject_data.get('band_cv', {})
    if band_cv:
        cv_values = [band_cv.get(b, 0) for b in BANDS if b in band_cv]
        if len(cv_values) >= 4:
            x = np.arange(len(cv_values))
            slope, _, r2, p, _ = stats.linregress(x, cv_values)
            results['cv_gradient'] = float(slope)  # CV change across band hierarchy
            results['cv_gradient_r2'] = float(r2)
            results['cv_hilo_diff'] = float(cv_values[-1] - cv_values[0]) if len(cv_values) >= 2 else 0
    
    # Band self-rate as "stickiness" proxy:
    # High self-rate = stays in same mode (viscous/sticky). Low = rapid switching.
    band_sr = subject_data.get('band_self_rates', {})
    if band_sr:
        sr_values = [band_sr.get(b, 0) for b in BANDS if b in band_sr]
        if len(sr_values) >= 4:
            x = np.arange(len(sr_values))
            slope, _, r2, p, _ = stats.linregress(x, sr_values)
            results['self_rate_gradient'] = float(slope)
            results['stickiness_hilo_diff'] = float(sr_values[-1] - sr_values[0]) if len(sr_values) >= 2 else 0
    
    # Band dwell as temporal scale proxy
    band_dwell = subject_data.get('band_mean_dwell', {})
    if band_dwell:
        dwell_values = [band_dwell.get(b, 0) for b in BANDS if b in band_dwell]
        if len(dwell_values) >= 4:
            # Log-transform since dwells span orders of magnitude
            log_dwells = [np.log(d + 1) for d in dwell_values]
            x = np.arange(len(log_dwells))
            slope, _, r2, p, _ = stats.linregress(x, log_dwells)
            results['dwell_gradient'] = float(slope)
    
    # Combined viscosity proxy
    # High viscosity = steep self-rate gradient (slow bands sticky, fast bands flexible)
    #                + high mean CV (critical dynamics)
    #                + steep Zipf (concentrated preferences)
    mean_cv = subject_data.get('mean_cv', 0)
    zipf_alpha = subject_data.get('zipf_alpha', 0)
    sr_grad = results.get('self_rate_gradient', 0)
    results['viscosity_proxy'] = float(abs(sr_grad) * mean_cv * zipf_alpha)
    
    return results


# ============================================================================
# 4. ANALYSIS & VISUALIZATION 
# ============================================================================

def analyze_from_json(json_path):
    """Load existing Φ-Dwell results and compute β-gradient metrics."""
    print(f"\n{'='*70}")
    print(f"  β-SIEVE × Φ-DWELL BRAIN VISCOSITY ANALYZER")
    print(f"  Applying grokking viscosity probe to brain eigenmode dynamics")
    print(f"{'='*70}\n")
    
    with open(json_path) as f:
        all_data = json.load(f)
    
    groups = {'AD': [], 'CN': [], 'FTD': []}
    group_map = {'A': 'AD', 'C': 'CN', 'F': 'FTD'}
    
    for sub_id, sub_data in all_data.items():
        group_code = sub_data.get('group', '')
        group = group_map.get(group_code, group_code)
        if group not in groups:
            continue
        
        beta_metrics = compute_beta_from_json(sub_data)
        beta_metrics['subject'] = sub_id
        beta_metrics['group'] = group
        beta_metrics['mmse'] = sub_data.get('mmse', None)
        
        # Carry forward key Φ-Dwell metrics
        for key in ['vocab_size', 'entropy', 'perplexity', 'mean_cv', 'zipf_alpha',
                     'top5_concentration', 'self_rate', 'criticality_fraction',
                     'band_cv', 'band_self_rates', 'band_mean_dwell']:
            if key in sub_data:
                beta_metrics[key] = sub_data[key]
        
        groups[group].append(beta_metrics)
    
    print(f"  Subjects: AD={len(groups['AD'])}, FTD={len(groups['FTD'])}, CN={len(groups['CN'])}")
    
    # ================================================================
    # STATISTICAL ANALYSIS
    # ================================================================
    
    # New β-gradient metrics to test
    beta_metrics_list = [
        'cv_gradient', 'cv_hilo_diff', 'self_rate_gradient', 
        'stickiness_hilo_diff', 'dwell_gradient', 'viscosity_proxy'
    ]
    
    # Also re-test original metrics for comparison
    phi_metrics_list = [
        'vocab_size', 'entropy', 'perplexity', 'mean_cv', 
        'zipf_alpha', 'top5_concentration'
    ]
    
    all_metrics = beta_metrics_list + phi_metrics_list
    
    print(f"\n  {'Metric':<28s} {'CN':>8s} {'AD':>8s} {'FTD':>8s}   {'KW-p':>7s}  {'AD-CN p':>8s} {'FTD-CN p':>8s}")
    print(f"  {'─'*90}")
    
    significant_results = []
    all_results = {}
    
    for metric in all_metrics:
        vals = {}
        for g in ['CN', 'AD', 'FTD']:
            vals[g] = [s[metric] for s in groups[g] if metric in s and s[metric] is not None]
        
        if not all(len(vals[g]) >= 5 for g in ['CN', 'AD', 'FTD']):
            continue
        
        # Kruskal-Wallis
        try:
            kw_stat, kw_p = stats.kruskal(vals['CN'], vals['AD'], vals['FTD'])
        except:
            continue
        
        # Pairwise Mann-Whitney
        try:
            u_ad, p_ad = stats.mannwhitneyu(vals['AD'], vals['CN'], alternative='two-sided')
            u_ftd, p_ftd = stats.mannwhitneyu(vals['FTD'], vals['CN'], alternative='two-sided')
            u_af, p_af = stats.mannwhitneyu(vals['AD'], vals['FTD'], alternative='two-sided')
        except:
            continue
        
        means = {g: np.mean(vals[g]) for g in ['CN', 'AD', 'FTD']}
        
        sig_marker = ' *' if kw_p < 0.05 else '  '
        is_new = '→' if metric in beta_metrics_list else ' '
        
        print(f" {is_new}{metric:<27s} {means['CN']:>8.4f} {means['AD']:>8.4f} {means['FTD']:>8.4f}  "
              f"{kw_p:>7.4f}{sig_marker} {p_ad:>8.4f}  {p_ftd:>8.4f}")
        
        all_results[metric] = {
            'means': means,
            'kw_p': kw_p,
            'pairwise': {'AD_vs_CN': p_ad, 'FTD_vs_CN': p_ftd, 'AD_vs_FTD': p_af},
            'is_new': metric in beta_metrics_list
        }
        
        if kw_p < 0.05:
            significant_results.append(metric)
        
        # MMSE correlation
        all_mmse = []
        all_vals = []
        for g in ['CN', 'AD', 'FTD']:
            for s in groups[g]:
                if metric in s and s[metric] is not None and s.get('mmse') is not None:
                    all_mmse.append(s['mmse'])
                    all_vals.append(s[metric])
        
        if len(all_mmse) > 20:
            rho, p_mmse = stats.spearmanr(all_mmse, all_vals)
            all_results[metric]['mmse_rho'] = float(rho)
            all_results[metric]['mmse_p'] = float(p_mmse)
    
    # ================================================================
    # MMSE CORRELATIONS for new metrics
    # ================================================================
    print(f"\n  MMSE CORRELATIONS (new β-gradient metrics)")
    print(f"  {'─'*50}")
    print(f"  {'Metric':<28s} {'Spearman ρ':>12s} {'p':>10s}")
    print(f"  {'─'*50}")
    
    for metric in beta_metrics_list:
        if metric in all_results and 'mmse_rho' in all_results[metric]:
            r = all_results[metric]
            sig = ' *' if r['mmse_p'] < 0.05 else '  '
            print(f"  {metric:<28s} {r['mmse_rho']:>12.3f} {r['mmse_p']:>10.4f}{sig}")
    
    # ================================================================
    # PER-BAND PROFILE COMPARISON
    # ================================================================
    print(f"\n  BAND HIERARCHY PROFILES (mean values by group)")
    print(f"  {'─'*70}")
    
    for profile_name, band_key in [('CV (criticality)', 'band_cv'), 
                                     ('Self-Rate (stickiness)', 'band_self_rates'),
                                     ('Log Mean Dwell', 'band_mean_dwell')]:
        print(f"\n  {profile_name}:")
        print(f"  {'Band':<8s}", end='')
        for g in ['CN', 'AD', 'FTD']:
            print(f"  {g:>8s}", end='')
        print()
        
        for band in BANDS:
            vals_by_group = {}
            for g in ['CN', 'AD', 'FTD']:
                band_vals = []
                for s in groups[g]:
                    bd = s.get(band_key, {})
                    if band in bd:
                        v = bd[band]
                        if profile_name.startswith('Log'):
                            v = np.log(v + 1)
                        band_vals.append(v)
                vals_by_group[g] = np.mean(band_vals) if band_vals else 0
            
            print(f"  {band:<8s}", end='')
            for g in ['CN', 'AD', 'FTD']:
                print(f"  {vals_by_group[g]:>8.3f}", end='')
            
            # Quick KW test per band
            band_data = {}
            for g in ['CN', 'AD', 'FTD']:
                band_data[g] = []
                for s in groups[g]:
                    bd = s.get(band_key, {})
                    if band in bd:
                        v = bd[band]
                        if profile_name.startswith('Log'):
                            v = np.log(v + 1)
                        band_data[g].append(v)
            try:
                _, p = stats.kruskal(band_data['CN'], band_data['AD'], band_data['FTD'])
                sig = ' *' if p < 0.05 else ''
                print(f"  p={p:.3f}{sig}", end='')
            except:
                pass
            print()
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('β-Sieve × Φ-Dwell: Brain Viscosity in Alzheimer\'s & FTD\n'
                     'OpenNeuro ds004504: AD (n=36) vs FTD (n=23) vs CN (n=29)',
                     fontsize=14, fontweight='bold')
        
        colors = {'CN': '#4DBEEE', 'AD': '#EDB120', 'FTD': '#D95319'}
        
        # Row 1: New β-gradient metrics
        plot_metrics_r1 = [
            ('cv_gradient', 'CV Gradient\n(band hierarchy slope)'),
            ('self_rate_gradient', 'Self-Rate Gradient\n(stickiness slope)'),
            ('viscosity_proxy', 'Viscosity Index\n(combined measure)'),
            ('cv_hilo_diff', 'CV Hi-Lo Diff\n(γ CV - δ CV)')
        ]
        
        for idx, (metric, title) in enumerate(plot_metrics_r1):
            ax = axes[0, idx]
            data_by_group = []
            labels = []
            for g in ['CN', 'AD', 'FTD']:
                vals = [s[metric] for s in groups[g] if metric in s]
                data_by_group.append(vals)
                labels.append(g)
            
            bp = ax.boxplot(data_by_group, labels=labels, patch_artist=True)
            for patch, g in zip(bp['boxes'], ['CN', 'AD', 'FTD']):
                patch.set_facecolor(colors[g])
                patch.set_alpha(0.6)
            
            # Overlay individual points
            for i, (g, vals) in enumerate(zip(['CN', 'AD', 'FTD'], data_by_group)):
                x_jitter = np.random.normal(i+1, 0.05, len(vals))
                ax.scatter(x_jitter, vals, c=colors[g], s=15, alpha=0.5, zorder=3)
            
            kw_p = all_results.get(metric, {}).get('kw_p', 1.0)
            sig = '*' if kw_p < 0.05 else 'ns'
            ax.set_title(f'{title}\nKW p={kw_p:.3f} ({sig})', fontsize=9)
        
        axes[0, 0].set_ylabel('NEW β-Gradient Metrics', fontsize=10, fontweight='bold')
        
        # Row 2: Original Φ-Dwell metrics for comparison
        plot_metrics_r2 = [
            ('vocab_size', 'Vocabulary Size'),
            ('mean_cv', 'Mean CV (Criticality)'),
            ('zipf_alpha', 'Zipf α'),
            ('top5_concentration', 'Top-5 Concentration')
        ]
        
        for idx, (metric, title) in enumerate(plot_metrics_r2):
            ax = axes[1, idx]
            data_by_group = []
            for g in ['CN', 'AD', 'FTD']:
                vals = [s[metric] for s in groups[g] if metric in s]
                data_by_group.append(vals)
            
            bp = ax.boxplot(data_by_group, labels=['CN', 'AD', 'FTD'], patch_artist=True)
            for patch, g in zip(bp['boxes'], ['CN', 'AD', 'FTD']):
                patch.set_facecolor(colors[g])
                patch.set_alpha(0.6)
            
            for i, (g, vals) in enumerate(zip(['CN', 'AD', 'FTD'], data_by_group)):
                x_jitter = np.random.normal(i+1, 0.05, len(vals))
                ax.scatter(x_jitter, vals, c=colors[g], s=15, alpha=0.5, zorder=3)
            
            kw_p = all_results.get(metric, {}).get('kw_p', 1.0)
            sig = '*' if kw_p < 0.05 else 'ns'
            ax.set_title(f'{title}\nKW p={kw_p:.3f} ({sig})', fontsize=9)
        
        axes[1, 0].set_ylabel('Original Φ-Dwell Metrics', fontsize=10, fontweight='bold')
        
        # Row 3: Band hierarchy profiles & MMSE correlations
        # 3a: CV profile across bands
        ax = axes[2, 0]
        for g in ['CN', 'AD', 'FTD']:
            means = []
            sems = []
            for band in BANDS:
                vals = [s['band_cv'][band] for s in groups[g] if 'band_cv' in s and band in s['band_cv']]
                means.append(np.mean(vals) if vals else 0)
                sems.append(np.std(vals)/np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ax.errorbar(range(5), means, yerr=sems, label=g, color=colors[g], 
                       linewidth=2, marker='o', capsize=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'])
        ax.set_ylabel('CV')
        ax.set_title('CV Profile Across Bands\n(slope = cv_gradient)', fontsize=9)
        ax.legend(fontsize=8)
        
        # 3b: Self-rate profile
        ax = axes[2, 1]
        for g in ['CN', 'AD', 'FTD']:
            means = []
            sems = []
            for band in BANDS:
                vals = [s['band_self_rates'][band] for s in groups[g] 
                        if 'band_self_rates' in s and band in s['band_self_rates']]
                means.append(np.mean(vals) if vals else 0)
                sems.append(np.std(vals)/np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ax.errorbar(range(5), means, yerr=sems, label=g, color=colors[g],
                       linewidth=2, marker='o', capsize=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'])
        ax.set_ylabel('Self-Rate')
        ax.set_title('Self-Rate Profile Across Bands\n(slope = self_rate_gradient)', fontsize=9)
        ax.legend(fontsize=8)
        
        # 3c: MMSE correlation scatter for best new metric
        ax = axes[2, 2]
        best_metric = None
        best_p = 1.0
        for m in beta_metrics_list:
            if m in all_results and 'mmse_p' in all_results[m]:
                if all_results[m]['mmse_p'] < best_p:
                    best_p = all_results[m]['mmse_p']
                    best_metric = m
        
        if best_metric:
            for g in ['CN', 'AD', 'FTD']:
                mmse_vals = [s['mmse'] for s in groups[g] if best_metric in s and s['mmse'] is not None]
                metric_vals = [s[best_metric] for s in groups[g] if best_metric in s and s['mmse'] is not None]
                ax.scatter(mmse_vals, metric_vals, c=colors[g], label=g, alpha=0.6, s=30)
            
            all_mmse = []
            all_vals = []
            for g in ['CN', 'AD', 'FTD']:
                for s in groups[g]:
                    if best_metric in s and s['mmse'] is not None:
                        all_mmse.append(s['mmse'])
                        all_vals.append(s[best_metric])
            
            if len(all_mmse) > 5:
                z = np.polyfit(all_mmse, all_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(all_mmse), max(all_mmse), 50)
                ax.plot(x_line, p(x_line), 'k--', alpha=0.5)
            
            rho = all_results[best_metric].get('mmse_rho', 0)
            ax.set_xlabel('MMSE Score')
            ax.set_ylabel(best_metric)
            ax.set_title(f'MMSE vs {best_metric}\nρ={rho:.3f}, p={best_p:.4f}', fontsize=9)
            ax.legend(fontsize=8)
        
        # 3d: Summary table
        ax = axes[2, 3]
        ax.axis('off')
        
        table_data = []
        headers = ['Metric', 'CN', 'AD', 'FTD', 'p']
        for m in beta_metrics_list:
            if m in all_results:
                r = all_results[m]
                sig = '*' if r['kw_p'] < 0.05 else ''
                table_data.append([
                    m[:20],
                    f"{r['means']['CN']:.4f}",
                    f"{r['means']['AD']:.4f}",
                    f"{r['means']['FTD']:.4f}",
                    f"{r['kw_p']:.3f}{sig}"
                ])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            ax.set_title('β-Gradient Metrics Summary', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        out_path = os.path.splitext(json_path)[0] + '_viscosity.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {out_path}")
        
        # Also save results JSON
        json_out = os.path.splitext(json_path)[0] + '_viscosity.json'
        
        # Compile per-subject results
        subject_results = {}
        for g in ['CN', 'AD', 'FTD']:
            for s in groups[g]:
                subject_results[s['subject']] = {
                    k: v for k, v in s.items() 
                    if k not in ['band_cv', 'band_self_rates', 'band_mean_dwell']
                }
        
        output = {
            'statistics': all_results,
            'significant_new_metrics': [m for m in beta_metrics_list if m in significant_results],
            'subjects': subject_results
        }
        
        with open(json_out, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved: {json_out}")
        
        return output, out_path
        
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
        return all_results, None


# ============================================================================
# 5. FULL EEG PIPELINE (for ds004504 .set files)
# ============================================================================

def run_full_pipeline(dataset_path):
    """Run complete β-Sieve × Φ-Dwell analysis on raw EEG files."""
    print(f"\n{'='*70}")
    print(f"  β-SIEVE × Φ-DWELL: FULL EEG PIPELINE")
    print(f"  Dataset: {dataset_path}")
    print(f"{'='*70}\n")
    
    try:
        import mne
    except ImportError:
        print("  ERROR: mne-python required for raw EEG processing.")
        print("  Install: pip install mne --break-system-packages")
        print("  Or use JSON mode: python brain_viscosity.py results.json")
        return
    
    electrode_names, eigenmodes, eigenvalues = build_eigenmodes(n_modes=6)
    print(f"  Eigenmodes: {len(eigenvalues)} modes, λ = {eigenvalues.round(2)}")
    
    # Find participants file
    participants_file = os.path.join(dataset_path, 'participants.tsv')
    if not os.path.exists(participants_file):
        print(f"  ERROR: {participants_file} not found")
        return
    
    # Parse participants
    subjects = {}
    with open(participants_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = dict(zip(header, parts))
            sub_id = row.get('participant_id', '')
            group = row.get('Group', row.get('group', ''))
            mmse = row.get('MMSE', row.get('mmse', ''))
            try:
                mmse = float(mmse)
            except:
                mmse = None
            subjects[sub_id] = {'group': group, 'mmse': mmse}
    
    print(f"  Participants: {len(subjects)}")
    
    all_results = {}
    
    for sub_id, sub_info in sorted(subjects.items()):
        # Find EEG file
        eeg_dir = os.path.join(dataset_path, sub_id, 'eeg')
        if not os.path.exists(eeg_dir):
            continue
        
        set_files = [f for f in os.listdir(eeg_dir) if f.endswith('.set')]
        if not set_files:
            continue
        
        set_path = os.path.join(eeg_dir, set_files[0])
        
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
            
            result = compute_brain_viscosity_from_eeg(
                data, sfreq, ch_names, electrode_names, eigenmodes
            )
            
            if result is not None:
                result['group'] = sub_info['group']
                result['mmse'] = sub_info['mmse']
                result['subject'] = sub_id
                all_results[sub_id] = result
                
                bg = result.get('beta_gradient', 0)
                vi = result.get('viscosity_index', 0)
                vocab = result.get('vocab_size', 0)
                
                print(f"  {sub_id} [{sub_info['group']}] MMSE={sub_info['mmse']} "
                      f"β-grad={bg:.4f} visc={vi:.4f} vocab={vocab}")
        
        except Exception as e:
            print(f"  {sub_id}: ERROR - {e}")
    
    # Save results
    out_json = os.path.join(dataset_path, 'brain_viscosity_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")
    
    # Run statistics on the results
    # Group the subjects
    groups = {'AD': [], 'CN': [], 'FTD': []}
    group_map = {'A': 'AD', 'C': 'CN', 'F': 'FTD'}
    
    for sub_id, r in all_results.items():
        g = group_map.get(r.get('group', ''), r.get('group', ''))
        if g in groups:
            groups[g].append(r)
    
    # Test new metrics
    new_metrics = ['beta_gradient', 'roughness_hilo_diff', 'roughness_coherence',
                   'spatial_beta_gradient', 'viscosity_index']
    
    print(f"\n  FULL PIPELINE RESULTS")
    print(f"  {'Metric':<25s} {'CN':>8s} {'AD':>8s} {'FTD':>8s}  {'KW-p':>7s}")
    print(f"  {'─'*60}")
    
    for metric in new_metrics + ['vocab_size', 'entropy', 'mean_cv', 'perplexity']:
        vals = {}
        for g in ['CN', 'AD', 'FTD']:
            vals[g] = [s[metric] for s in groups[g] if metric in s]
        
        if not all(len(v) >= 3 for v in vals.values()):
            continue
        
        try:
            _, kw_p = stats.kruskal(vals['CN'], vals['AD'], vals['FTD'])
        except:
            continue
        
        means = {g: np.mean(v) for g, v in vals.items()}
        sig = ' *' if kw_p < 0.05 else ''
        is_new = '→' if metric in new_metrics else ' '
        print(f" {is_new}{metric:<24s} {means['CN']:>8.4f} {means['AD']:>8.4f} {means['FTD']:>8.4f}  {kw_p:>7.4f}{sig}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python brain_viscosity.py phidwell_alzheimer_results.json")
        print("  python brain_viscosity.py /path/to/ds004504/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if path.endswith('.json'):
        analyze_from_json(path)
    elif os.path.isdir(path):
        # Check if there's a JSON in the directory
        json_candidates = [f for f in os.listdir(path) if 'viscosity' in f and f.endswith('.json')]
        if '--full' in sys.argv or not json_candidates:
            run_full_pipeline(path)
        else:
            analyze_from_json(os.path.join(path, json_candidates[0]))
    else:
        print(f"  ERROR: {path} not found")
        sys.exit(1)
