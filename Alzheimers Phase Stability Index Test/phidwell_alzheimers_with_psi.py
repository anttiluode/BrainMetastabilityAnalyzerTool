#!/usr/bin/env python3
"""
Φ-Dwell Alzheimer's Eigenmode Analyzer
========================================
Applies eigenmode vocabulary analysis to the OpenNeuro ds004504 dataset:
  36 Alzheimer's disease (AD), 23 Frontotemporal dementia (FTD), 29 Healthy controls (CN)
  19-channel 10-20 EEG, resting state eyes-closed, 500 Hz

For each subject, computes:
  - Eigenmode vocabulary size (number of distinct cross-band configurations)
  - Shannon entropy of the word distribution
  - Self-transition rate (how repetitive the dynamics are)
  - Bigram perplexity (how predictable the sequence is)
  - Cross-band eigenmode coupling matrix
  - Criticality metrics (CV, regime classification)
  - Per-band self-transition rates
  - Zipf exponent of the word frequency distribution

Then tests whether these metrics separate AD, FTD, and CN groups,
and correlates with MMSE scores.

Dataset: https://openneuro.org/datasets/ds004504
  Download: openneuro download --snapshot 1.0.8 ds004504 ds004504/
  Or: wget the zip from https://nemar.org/dataexplorer/detail?dataset_id=ds004504

Usage:
    python phidwell_alzheimer.py "path/to/ds004504/"
    python phidwell_alzheimer.py "path/to/ds004504/" --use-derivatives

Requirements:
    pip install numpy scipy matplotlib mne scikit-learn
"""

import numpy as np
import scipy.signal
import scipy.linalg
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
import argparse
import sys
import os
import glob
import json
import csv
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
except ImportError:
    print("ERROR: mne required. pip install mne")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 50),
}
BAND_NAMES = list(BANDS.keys())
N_MODES = 6  # 19 channels → ~6 usable eigenmodes (skip trivial mode 0)

MODE_NAMES = ['A-P', 'L-R', 'C-P', 'Diag', 'M5', 'M6']

# 19-channel 10-20 system electrode positions
# The ds004504 uses: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
ELECTRODE_POS_19 = {
    'Fp1': (-0.30, 0.90), 'Fp2': (0.30, 0.90),
    'F7':  (-0.70, 0.60), 'F3':  (-0.35, 0.60), 'Fz': (0.00, 0.60),
    'F4':  (0.35, 0.60),  'F8':  (0.70, 0.60),
    'T3':  (-0.90, 0.00), 'C3':  (-0.40, 0.00), 'Cz': (0.00, 0.00),
    'C4':  (0.40, 0.00),  'T4':  (0.90, 0.00),
    'T5':  (-0.70, -0.50),'P3':  (-0.35, -0.50),'Pz': (0.00, -0.50),
    'P4':  (0.35, -0.50), 'T6':  (0.70, -0.50),
    'O1':  (-0.30, -0.85),'O2':  (0.30, -0.85),
}

# Alternative channel names that might appear in the dataset
CHANNEL_ALIASES = {
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    'TP7': 'T5', 'TP8': 'T6',
    'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2', 'EEG F7': 'F7',
    'EEG F3': 'F3', 'EEG Fz': 'Fz', 'EEG F4': 'F4', 'EEG F8': 'F8',
    'EEG T3': 'T3', 'EEG C3': 'C3', 'EEG Cz': 'Cz', 'EEG C4': 'C4',
    'EEG T4': 'T4', 'EEG T5': 'T5', 'EEG P3': 'P3', 'EEG Pz': 'Pz',
    'EEG P4': 'P4', 'EEG T6': 'T6', 'EEG O1': 'O1', 'EEG O2': 'O2',
}



# ═══════════════════════════════════════════════════════════════
# PHASE-STABILITY INDEX (PSI) — New biomarker
# ═══════════════════════════════════════════════════════════════
# For each time window's eigenmode power vector, we run iterative
# Gerchberg-Saxton phase recovery and measure convergence rate.
# Fast convergence = coherent eigenmode configuration.
# Slow convergence = ambiguous/incoherent configuration.

PSI_CANVAS = 32
PSI_MAX_ITERS = 20
PSI_EPSILON = 0.05
PSI_INERTIA = 0.85


def compute_phase_stability_index(coeffs_ds, window_stride=4, n_windows=None):
    n_bands, n_steps, n_modes = coeffs_ds.shape
    s = PSI_CANVAS
    center = s / 2.0

    y, x = np.ogrid[:s, :s]
    r_grid = np.sqrt((x - center)**2 + (y - center)**2)
    r_max = center

    last_phase = None
    psi_iters = []
    psi_residuals = []
    convergence_curves = []
    band_psi = {b: [] for b in BAND_NAMES}

    total_windows = (n_steps - 1) // window_stride
    if n_windows is not None:
        total_windows = min(total_windows, n_windows)

    for wi in range(total_windows):
        t = wi * window_stride
        if t >= n_steps:
            break

        mode_powers = coeffs_ds[:, t, :]
        profile_len = s // 2
        radial_profile = np.zeros(profile_len, dtype=np.float32)

        for bi in range(n_bands):
            band_power = mode_powers[bi]
            r_start = int(bi * profile_len / n_bands)
            r_end = int((bi + 1) * profile_len / n_bands)
            for mi in range(min(n_modes, r_end - r_start)):
                r_idx = r_start + mi
                if r_idx < profile_len:
                    radial_profile[r_idx] = band_power[mi]

        rp_max = np.max(radial_profile)
        if rp_max > 1e-9:
            radial_profile /= rp_max

        r_norm = r_grid / r_max
        r_idx_map = np.clip((r_norm * (profile_len - 1)).astype(np.int32), 0, profile_len - 1)
        magnitude_2d = radial_profile[r_idx_map]
        mag_freq = np.abs(np.fft.fftshift(np.fft.fft2(magnitude_2d)))

        if last_phase is not None and last_phase.shape == mag_freq.shape:
            noise_scale = (1.0 - PSI_INERTIA) * 2.0
            phase = last_phase + np.random.uniform(
                -noise_scale, noise_scale, mag_freq.shape).astype(np.float32)
        else:
            phase = np.random.uniform(-np.pi, np.pi, mag_freq.shape).astype(np.float32)

        converged_at = PSI_MAX_ITERS
        curve = []

        for it in range(PSI_MAX_ITERS):
            prev_phase = phase.copy()
            spectrum = mag_freq * np.exp(1j * phase)
            spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum)))
            spatial = np.clip(spatial, 0, None)
            new_spectrum = np.fft.fftshift(np.fft.fft2(spatial))
            phase = np.angle(new_spectrum).astype(np.float32)

            phase_diff = np.mean(np.abs(phase - prev_phase))
            curve.append(float(phase_diff))

            if phase_diff < PSI_EPSILON and converged_at == PSI_MAX_ITERS:
                converged_at = it + 1

        last_phase = phase.copy()
        psi_iters.append(converged_at)
        psi_residuals.append(curve[-1] if curve else 0)
        convergence_curves.append(curve)

        for bi, bname in enumerate(BAND_NAMES):
            bp = mode_powers[bi]
            bp_norm = bp / (np.sum(bp) + 1e-9)
            band_ent = -np.sum(bp_norm * np.log2(bp_norm + 1e-15))
            band_psi[bname].append(band_ent)

    psi_iters = np.array(psi_iters, dtype=np.float64)
    psi_residuals = np.array(psi_residuals, dtype=np.float64)

    if convergence_curves:
        max_len = max(len(c) for c in convergence_curves)
        padded = np.zeros((len(convergence_curves), max_len))
        for i, c in enumerate(convergence_curves):
            padded[i, :len(c)] = c
        avg_curve = np.mean(padded, axis=0)
    else:
        avg_curve = np.zeros(PSI_MAX_ITERS)

    band_psi_means = {}
    for bname in BAND_NAMES:
        band_psi_means[bname] = float(np.mean(band_psi[bname])) if band_psi[bname] else 0.0

    bvals = [band_psi_means.get(b, 0) for b in BAND_NAMES]
    psi_gradient = float(np.polyfit(range(len(bvals)), bvals, 1)[0]) if len(bvals) >= 3 else 0.0

    return {
        'psi_mean': float(np.mean(psi_iters)) if len(psi_iters) > 0 else float(PSI_MAX_ITERS),
        'psi_std': float(np.std(psi_iters)) if len(psi_iters) > 0 else 0.0,
        'psi_median': float(np.median(psi_iters)) if len(psi_iters) > 0 else float(PSI_MAX_ITERS),
        'psi_residual_mean': float(np.mean(psi_residuals)) if len(psi_residuals) > 0 else 0.0,
        'psi_convergence_rate': float(avg_curve[0] - avg_curve[-1]) if len(avg_curve) > 1 else 0.0,
        'psi_band_entropy': band_psi_means,
        'psi_gradient': psi_gradient,
        'psi_n_windows': len(psi_iters),
        'psi_fraction_converged': float(np.mean(psi_iters < PSI_MAX_ITERS)) if len(psi_iters) > 0 else 0.0,
    }

# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN
# ═══════════════════════════════════════════════════════════════

def build_graph_laplacian(positions, sigma=0.5):
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            w = np.exp(-d ** 2 / (2 * sigma ** 2))
            A[i, j] = A[j, i] = w
    D = np.diag(A.sum(axis=1))
    L = D - A
    vals, vecs = scipy.linalg.eigh(L)
    return names, coords, vecs[:, 1:N_MODES + 1], vals[1:N_MODES + 1]


def map_channels(raw_names, graph_names):
    """Map EEG channel names to graph electrode names. Handles various naming conventions."""
    mapping = {}
    graph_lower = {n.lower(): n for n in graph_names}
    
    for ch in raw_names:
        # Clean the channel name
        clean = ch.strip()
        
        # Direct match
        if clean in graph_names:
            mapping[ch] = clean
            continue
        
        # Case-insensitive match
        if clean.lower() in graph_lower:
            mapping[ch] = graph_lower[clean.lower()]
            continue
        
        # Check aliases
        if clean in CHANNEL_ALIASES:
            alias = CHANNEL_ALIASES[clean]
            if alias in graph_names:
                mapping[ch] = alias
                continue
        
        # Strip 'EEG ' prefix and try again
        stripped = clean.replace('EEG ', '').replace('EEG-', '').strip()
        if stripped in graph_names:
            mapping[ch] = stripped
            continue
        if stripped.lower() in graph_lower:
            mapping[ch] = graph_lower[stripped.lower()]
            continue
        if stripped in CHANNEL_ALIASES:
            alias = CHANNEL_ALIASES[stripped]
            if alias in graph_names:
                mapping[ch] = alias
                continue
        
        # Strip dots (PhysioNet style)
        nodots = stripped.replace('.', '')
        if nodots in graph_names:
            mapping[ch] = nodots
            continue
        if nodots.lower() in graph_lower:
            mapping[ch] = graph_lower[nodots.lower()]
            continue
    
    return mapping


# ═══════════════════════════════════════════════════════════════
# EEG → EIGENMODE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_subject(filepath, graph_names, eigenvecs, word_step_ms=25,
                    max_duration_s=120):
    """
    Full eigenmode analysis for a single subject.
    Returns dict with all metrics, or None on failure.
    """
    try:
        # Try .set format first (EEGLAB), fall back to .edf
        if filepath.endswith('.set'):
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='error')
        else:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='error')
    except Exception as e:
        print(f"    ERROR reading {filepath}: {e}")
        return None
    
    mapping = map_channels(raw.ch_names, graph_names)
    
    if len(mapping) < 10:
        print(f"    Only {len(mapping)}/{len(graph_names)} channels mapped, skipping")
        return None
    
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samp = raw.n_times
    n_elec = len(graph_names)
    
    # Limit duration to avoid memory issues on long recordings
    max_samp = int(max_duration_s * sfreq)
    if n_samp > max_samp:
        data = data[:, :max_samp]
        n_samp = max_samp
    
    duration_s = n_samp / sfreq
    
    # Band-filter and extract phases
    phases = np.zeros((5, n_elec, n_samp), dtype=np.complex64)
    
    for bi, band in enumerate(BAND_NAMES):
        lo, hi = BANDS[band]
        # Ensure valid filter range
        if hi >= sfreq / 2:
            hi = sfreq / 2 - 1
        if lo >= hi:
            continue
        
        sos = scipy.signal.butter(3, [lo, hi], btype='band', fs=sfreq, output='sos')
        
        for raw_ch, graph_ch in mapping.items():
            idx_g = graph_names.index(graph_ch)
            idx_r = raw.ch_names.index(raw_ch)
            sig = scipy.signal.sosfiltfilt(sos, data[idx_r])
            analytic = scipy.signal.hilbert(sig)
            phases[bi, idx_g, :] = analytic / (np.abs(analytic) + 1e-9)
    
    # Project onto eigenmodes
    coeffs = np.abs(np.tensordot(phases, eigenvecs, axes=([1], [0])))
    # coeffs shape: (5, n_samp, N_MODES)
    
    # Dominant mode per band per timestep
    tokens = np.argmax(coeffs, axis=2)  # (5, n_samp)
    
    # Downsample to word_step
    step = max(1, int(sfreq * word_step_ms / 1000))
    tokens_ds = tokens[:, ::step]
    coeffs_ds = coeffs[:, ::step, :]
    
    # Words
    n_words = tokens_ds.shape[1]
    words = [tuple(tokens_ds[:, t]) for t in range(n_words)]
    
    # ── Vocabulary metrics ──
    counts = Counter(words)
    vocab_size = len(counts)
    theoretical = N_MODES ** 5
    probs = np.array(list(counts.values())) / n_words
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    
    # Zipf
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    if len(freqs) > 5:
        n_fit = min(50, len(freqs))
        slope, _, r, _, _ = stats.linregress(
            np.log(ranks[:n_fit]), np.log(freqs[:n_fit]))
        zipf_alpha = -slope
        zipf_r2 = r ** 2
    else:
        zipf_alpha = 0
        zipf_r2 = 0
    
    # ── Transition metrics ──
    # Self-transition rate (global)
    n_self = sum(1 for a, b in zip(words[:-1], words[1:]) if a == b)
    self_rate = n_self / (n_words - 1) if n_words > 1 else 0
    
    # Per-band self-transition rate
    band_self_rates = {}
    for bi, band in enumerate(BAND_NAMES):
        band_tokens = tokens_ds[bi, :]
        n_band_self = np.sum(band_tokens[:-1] == band_tokens[1:])
        band_self_rates[band] = float(n_band_self / (len(band_tokens) - 1))
    
    # ── Bigram perplexity ──
    # Build bigram model
    bigram_counts = defaultdict(Counter)
    context_totals = defaultdict(int)
    unigram_counts = Counter(words)
    
    for i in range(len(words) - 1):
        context = (words[i],)
        target = words[i + 1]
        bigram_counts[context][target] += 1
        context_totals[context] += 1
    
    # Compute perplexity
    V = max(vocab_size, 1)
    k = 0.01
    total_log_prob = 0
    n_scored = 0
    
    for i in range(1, len(words)):
        context = (words[i - 1],)
        word = words[i]
        if context in bigram_counts:
            count = bigram_counts[context].get(word, 0)
            total = context_totals[context]
            prob = (count + k) / (total + k * V)
        else:
            count = unigram_counts.get(word, 0)
            prob = (count + k) / (n_words + k * V)
        total_log_prob += np.log2(prob + 1e-15)
        n_scored += 1
    
    perplexity = 2.0 ** (-total_log_prob / n_scored) if n_scored > 0 else float('inf')
    
    # ── Cross-band coupling ──
    # Correlation of dominant mode across bands
    coupling_matrix = np.zeros((5, 5))
    for bi in range(5):
        for bj in range(bi, 5):
            if bi == bj:
                coupling_matrix[bi, bj] = 1.0
            else:
                # Correlation of dominant mode index timeseries
                r, _ = stats.pearsonr(tokens_ds[bi].astype(float),
                                       tokens_ds[bj].astype(float))
                coupling_matrix[bi, bj] = coupling_matrix[bj, bi] = r if np.isfinite(r) else 0
    
    # Key coupling metrics
    delta_theta_coupling = coupling_matrix[0, 1]
    alpha_beta_coupling = coupling_matrix[2, 3]
    mean_coupling = np.mean(coupling_matrix[np.triu_indices(5, k=1)])
    
    # ── Dwell time / criticality ──
    # For each band, compute dwell times of dominant mode
    band_cv = {}
    band_mean_dwell = {}
    for bi, band in enumerate(BAND_NAMES):
        bt = tokens_ds[bi, :]
        # Dwell = consecutive runs of same mode
        dwells = []
        current_len = 1
        for t in range(1, len(bt)):
            if bt[t] == bt[t - 1]:
                current_len += 1
            else:
                dwells.append(current_len * word_step_ms)
                current_len = 1
        dwells.append(current_len * word_step_ms)
        
        dwells = np.array(dwells, dtype=float)
        if len(dwells) > 2:
            band_cv[band] = float(np.std(dwells) / (np.mean(dwells) + 1e-9))
            band_mean_dwell[band] = float(np.mean(dwells))
        else:
            band_cv[band] = 0
            band_mean_dwell[band] = 0
    
    # Global criticality: fraction of bands with CV > 1.0
    n_critical = sum(1 for v in band_cv.values() if v > 1.0)
    criticality_fraction = n_critical / 5
    mean_cv = np.mean(list(band_cv.values()))
    
    # ── Top words ──
    top_words = counts.most_common(10)
    top_word_strs = [''.join(str(x + 1) for x in w) for w, _ in top_words]
    top_word_fracs = [c / n_words for _, c in top_words]
    
    # Concentration: fraction in top-5 words
    top5_frac = sum(c for _, c in counts.most_common(5)) / n_words
    
    # ── Phase-Stability Index ──
    psi_metrics = compute_phase_stability_index(
        coeffs_ds, window_stride=4, n_windows=500)
    
    result = {
        'n_channels': len(mapping),
        'duration_s': duration_s,
        'n_words': n_words,
        'vocab_size': vocab_size,
        'vocab_usage_pct': vocab_size / theoretical * 100,
        'entropy': entropy,
        'self_rate': self_rate,
        'perplexity': perplexity,
        'zipf_alpha': zipf_alpha,
        'zipf_r2': zipf_r2,
        'top5_concentration': top5_frac,
        'delta_theta_coupling': delta_theta_coupling,
        'alpha_beta_coupling': alpha_beta_coupling,
        'mean_coupling': mean_coupling,
        'criticality_fraction': criticality_fraction,
        'mean_cv': mean_cv,
        'band_self_rates': band_self_rates,
        'band_cv': band_cv,
        'band_mean_dwell': band_mean_dwell,
        'coupling_matrix': coupling_matrix.tolist(),
        'top_words': top_word_strs[:5],
        'top_word_fracs': top_word_fracs[:5],
    }
    result.update(psi_metrics)
    return result


# ═══════════════════════════════════════════════════════════════
# DATASET PARSING
# ═══════════════════════════════════════════════════════════════

def parse_participants(ds_path):
    """Parse participants.tsv to get group labels and MMSE scores."""
    tsv_path = os.path.join(ds_path, 'participants.tsv')
    if not os.path.exists(tsv_path):
        print(f"WARNING: {tsv_path} not found")
        return {}
    
    participants = {}
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            subj_id = row.get('participant_id', '').strip()
            group = row.get('Group', row.get('group', '')).strip()
            
            # MMSE score
            mmse_str = row.get('MMSE', row.get('mmse', '')).strip()
            try:
                mmse = float(mmse_str) if mmse_str and mmse_str != 'n/a' else None
            except ValueError:
                mmse = None
            
            # Age
            age_str = row.get('age', '').strip()
            try:
                age = float(age_str) if age_str and age_str != 'n/a' else None
            except ValueError:
                age = None
            
            # Gender
            gender = row.get('sex', row.get('Gender', row.get('gender', ''))).strip()
            
            participants[subj_id] = {
                'group': group,
                'mmse': mmse,
                'age': age,
                'gender': gender,
            }
    
    return participants


def find_eeg_files(ds_path, use_derivatives=False):
    """Find all EEG files in the BIDS dataset."""
    if use_derivatives:
        search_base = os.path.join(ds_path, 'derivatives')
    else:
        search_base = ds_path
    
    # Look for .set files (EEGLAB format, standard for this dataset)
    files = sorted(glob.glob(os.path.join(search_base, 'sub-*', 'eeg', '*.set')))
    
    if not files:
        # Try without eeg subfolder
        files = sorted(glob.glob(os.path.join(search_base, 'sub-*', '*.set')))
    
    if not files:
        # Try .edf
        files = sorted(glob.glob(os.path.join(search_base, 'sub-*', 'eeg', '*.edf')))
    
    if not files:
        files = sorted(glob.glob(os.path.join(search_base, 'sub-*', '*.edf')))
    
    if not files:
        # Try .fif
        files = sorted(glob.glob(os.path.join(search_base, 'sub-*', 'eeg', '*eeg.fif')))
    
    # Map to subject IDs
    file_map = {}
    for f in files:
        # Extract subject ID from path
        parts = f.replace('\\', '/').split('/')
        for p in parts:
            if p.startswith('sub-'):
                file_map[p] = f
                break
    
    return file_map


# ═══════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════

def group_comparison(results, metric_name):
    """Compare a metric across AD, FTD, CN groups."""
    groups = {'A': [], 'F': [], 'C': []}
    group_labels = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
    
    for subj_id, data in results.items():
        g = data['group']
        val = data['metrics'].get(metric_name)
        if val is not None and np.isfinite(val):
            if g in groups:
                groups[g].append(val)
    
    out = {}
    for g, label in group_labels.items():
        vals = groups[g]
        if vals:
            out[label] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'n': len(vals),
                'values': vals,
            }
    
    # Kruskal-Wallis test (non-parametric, works with unequal sizes)
    group_vals = [groups[g] for g in ['A', 'F', 'C'] if groups[g]]
    if len(group_vals) >= 2 and all(len(g) >= 3 for g in group_vals):
        h_stat, kw_p = stats.kruskal(*group_vals)
        out['kruskal_H'] = h_stat
        out['kruskal_p'] = kw_p
    else:
        out['kruskal_H'] = 0
        out['kruskal_p'] = 1.0
    
    # Pairwise Mann-Whitney tests
    for g1, g2 in [('A', 'C'), ('F', 'C'), ('A', 'F')]:
        label = f'{group_labels[g1]}_vs_{group_labels[g2]}'
        if groups[g1] and groups[g2] and len(groups[g1]) >= 3 and len(groups[g2]) >= 3:
            u_stat, mw_p = stats.mannwhitneyu(groups[g1], groups[g2], alternative='two-sided')
            # Effect size (rank-biserial)
            n1, n2 = len(groups[g1]), len(groups[g2])
            effect = 1 - (2 * u_stat) / (n1 * n2)
            out[label] = {'U': u_stat, 'p': mw_p, 'effect': effect}
        else:
            out[label] = {'U': 0, 'p': 1.0, 'effect': 0}
    
    return out


def mmse_correlation(results, metric_name):
    """Correlate a metric with MMSE scores across all subjects."""
    vals = []
    mmse_scores = []
    
    for subj_id, data in results.items():
        mmse = data.get('mmse')
        val = data['metrics'].get(metric_name)
        if mmse is not None and val is not None and np.isfinite(val) and np.isfinite(mmse):
            vals.append(val)
            mmse_scores.append(mmse)
    
    if len(vals) < 5:
        return {'r': 0, 'p': 1.0, 'n': len(vals)}
    
    r, p = stats.pearsonr(mmse_scores, vals)
    rho, rho_p = stats.spearmanr(mmse_scores, vals)
    
    return {
        'pearson_r': r, 'pearson_p': p,
        'spearman_rho': rho, 'spearman_p': rho_p,
        'n': len(vals),
        'mmse_values': mmse_scores,
        'metric_values': vals,
    }


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_alzheimer_dashboard(results, save_path):
    """Main results dashboard."""
    
    fig = plt.figure(figsize=(22, 24))
    fig.suptitle("Φ-Dwell Alzheimer's Eigenmode Analysis + Phase-Stability Index\n"
                 "OpenNeuro ds004504: AD (n=36) vs FTD (n=23) vs CN (n=29)",
                 fontsize=16, fontweight='bold', y=0.99)
    
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    colors = {'AD': '#D94A4A', 'FTD': '#E8A838', 'CN': '#4A90D9'}
    
    # Collect group data
    groups = {'AD': 'A', 'FTD': 'F', 'CN': 'C'}
    
    def get_group_vals(metric):
        out = {}
        for label, code in groups.items():
            vals = [r['metrics'][metric] for r in results.values()
                    if r['group'] == code and r['metrics'].get(metric) is not None
                    and np.isfinite(r['metrics'].get(metric, float('nan')))]
            out[label] = vals
        return out
    
    # ── Key metrics to test ──
    key_metrics = [
        ('vocab_size', 'Vocabulary Size', 'words'),
        ('entropy', 'Shannon Entropy', 'bits'),
        ('perplexity', 'Bigram Perplexity', ''),
        ('self_rate', 'Self-Transition Rate', ''),
        ('mean_cv', 'Mean CV (Criticality)', ''),
        ('delta_theta_coupling', 'δ-θ Coupling', 'r'),
        ('mean_coupling', 'Mean Cross-Band Coupling', 'r'),
        ('top5_concentration', 'Top-5 Concentration', ''),
        ('psi_mean', 'PSI Mean Iterations', 'iters'),
        ('psi_gradient', 'PSI Band Gradient', ''),
        ('psi_fraction_converged', 'PSI Fraction Converged', ''),
        ('psi_residual_mean', 'PSI Residual', 'rad'),
    ]
    
    # ── Panels 1-12: Box plots for each metric ──
    positions_map = [(0, 0), (0, 1), (0, 2), (0, 3),
                     (1, 0), (1, 1), (1, 2), (1, 3),
                     (2, 0), (2, 1), (2, 2), (2, 3)]
    
    for idx, (metric, title, unit) in enumerate(key_metrics):
        row, col = positions_map[idx]
        ax = fig.add_subplot(gs[row, col])
        
        gv = get_group_vals(metric)
        comp = group_comparison(results, metric)
        
        # Box plot
        bp_data = [gv.get('CN', []), gv.get('AD', []), gv.get('FTD', [])]
        bp_labels = ['CN', 'AD', 'FTD']
        bp_colors = [colors['CN'], colors['AD'], colors['FTD']]
        
        bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Overlay individual points
        for i, (data, color) in enumerate(zip(bp_data, bp_colors)):
            x = np.random.normal(i + 1, 0.08, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=15, color=color, edgecolors='none')
        
        # Stats annotation
        kw_p = comp.get('kruskal_p', 1.0)
        if kw_p < 0.001:
            sig_str = f'KW p < 0.001'
        elif kw_p < 0.05:
            sig_str = f'KW p = {kw_p:.3f}'
        else:
            sig_str = f'KW p = {kw_p:.2f} (ns)'
        
        # Pairwise significance markers
        ad_cn = comp.get('AD_vs_CN', {}).get('p', 1.0)
        ftd_cn = comp.get('FTD_vs_CN', {}).get('p', 1.0)
        
        stars = ''
        if ad_cn < 0.001:
            stars += 'AD-CN ***  '
        elif ad_cn < 0.01:
            stars += 'AD-CN **  '
        elif ad_cn < 0.05:
            stars += 'AD-CN *  '
        
        if ftd_cn < 0.001:
            stars += 'FTD-CN ***'
        elif ftd_cn < 0.01:
            stars += 'FTD-CN **'
        elif ftd_cn < 0.05:
            stars += 'FTD-CN *'
        
        ax.set_title(f'{title}\n{sig_str}', fontsize=9, fontweight='bold')
        if stars:
            ax.text(0.5, 0.02, stars, transform=ax.transAxes, fontsize=7,
                    ha='center', color='darkred', fontweight='bold')
        if unit:
            ax.set_ylabel(unit, fontsize=8)
    
    # ── Panel: MMSE correlation scatter ──
    ax9 = fig.add_subplot(gs[3, 0])
    
    # Pick the best-correlating metric
    best_metric = 'vocab_size'
    best_r = 0
    for metric, _, _ in key_metrics:
        mc = mmse_correlation(results, metric)
        if abs(mc.get('spearman_rho', 0)) > abs(best_r):
            best_r = mc.get('spearman_rho', 0)
            best_metric = metric
    
    mc = mmse_correlation(results, best_metric)
    if mc['n'] > 5:
        for subj_id, data in results.items():
            mmse = data.get('mmse')
            val = data['metrics'].get(best_metric)
            if mmse is not None and val is not None and np.isfinite(val):
                g = data['group']
                label = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}.get(g, 'Other')
                ax9.scatter(mmse, val, color=colors.get(label, 'gray'),
                           alpha=0.6, s=30, edgecolors='none')
        
        # Regression line
        x = np.array(mc['mmse_values'])
        y = np.array(mc['metric_values'])
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(x.min(), x.max(), 100)
        ax9.plot(x_range, p_line(x_range), 'k--', alpha=0.5, linewidth=1)
        
        ax9.set_xlabel('MMSE Score', fontsize=9)
        ax9.set_ylabel(best_metric.replace('_', ' ').title(), fontsize=9)
        ax9.set_title(f'MMSE vs {best_metric}\n'
                      f'ρ={mc["spearman_rho"]:.3f}, p={mc["spearman_p"]:.4f}',
                      fontsize=9, fontweight='bold')
        
        # Legend
        for label, color in colors.items():
            ax9.scatter([], [], color=color, label=label, s=30)
        ax9.legend(fontsize=7, loc='best')
    
    # ── Panel: Per-band self-transition by group ──
    ax10 = fig.add_subplot(gs[3, 1])
    
    band_data = {label: {band: [] for band in BAND_NAMES} for label in ['AD', 'FTD', 'CN']}
    for subj_id, data in results.items():
        g = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}.get(data['group'], None)
        if g and data['metrics'].get('band_self_rates'):
            for band in BAND_NAMES:
                val = data['metrics']['band_self_rates'].get(band)
                if val is not None:
                    band_data[g][band].append(val)
    
    x = np.arange(len(BAND_NAMES))
    width = 0.25
    for i, (label, color) in enumerate([('CN', colors['CN']), ('AD', colors['AD']), ('FTD', colors['FTD'])]):
        means = [np.mean(band_data[label][b]) if band_data[label][b] else 0 for b in BAND_NAMES]
        sems = [np.std(band_data[label][b]) / np.sqrt(len(band_data[label][b]))
                if len(band_data[label][b]) > 1 else 0 for b in BAND_NAMES]
        ax10.bar(x + i * width, means, width, yerr=sems, label=label,
                 color=color, alpha=0.7, capsize=2)
    
    ax10.set_xticks(x + width)
    ax10.set_xticklabels([b.capitalize() for b in BAND_NAMES], fontsize=8)
    ax10.set_ylabel('Self-Transition Rate', fontsize=8)
    ax10.set_title('Per-Band Self-Transition by Group', fontsize=9, fontweight='bold')
    ax10.legend(fontsize=7)
    
    # ── Panel: MMSE correlation table ──
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')
    
    header = f"{'Metric':<22} {'ρ':>6} {'p':>8}"
    ax11.text(0.02, 0.95, header, fontsize=8, family='monospace',
              fontweight='bold', transform=ax11.transAxes)
    ax11.text(0.02, 0.90, '─' * 40, fontsize=7, family='monospace',
              transform=ax11.transAxes, color='gray')
    
    y = 0.84
    correlations = []
    for metric, title, _ in key_metrics:
        mc = mmse_correlation(results, metric)
        rho = mc.get('spearman_rho', 0)
        p = mc.get('spearman_p', 1.0)
        correlations.append((metric, title, rho, p))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for metric, title, rho, p in correlations:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        color = 'darkred' if p < 0.05 else 'black'
        line = f"{title:<22} {rho:>6.3f} {p:>8.4f} {sig}"
        ax11.text(0.02, y, line, fontsize=7.5, family='monospace',
                  transform=ax11.transAxes, color=color)
        y -= 0.075
    
    ax11.set_title('MMSE Correlations (Spearman)', fontsize=9, fontweight='bold')
    
    # ── Panel: Summary statistics table ──
    ax12 = fig.add_subplot(gs[3, 3])
    ax12.axis('off')
    
    header = f"{'Metric':<18} {'CN':>8} {'AD':>8} {'FTD':>8}"
    ax12.text(0.02, 0.95, header, fontsize=8, family='monospace',
              fontweight='bold', transform=ax12.transAxes)
    ax12.text(0.02, 0.90, '─' * 45, fontsize=7, family='monospace',
              transform=ax12.transAxes, color='gray')
    
    y = 0.84
    for metric, title, _ in key_metrics[:8]:
        gv = get_group_vals(metric)
        cn_m = np.mean(gv['CN']) if gv['CN'] else 0
        ad_m = np.mean(gv['AD']) if gv['AD'] else 0
        ftd_m = np.mean(gv['FTD']) if gv['FTD'] else 0
        
        # Format based on magnitude
        if cn_m > 100:
            line = f"{title[:18]:<18} {cn_m:>7.0f}  {ad_m:>7.0f}  {ftd_m:>7.0f}"
        else:
            line = f"{title[:18]:<18} {cn_m:>7.3f}  {ad_m:>7.3f}  {ftd_m:>7.3f}"
        
        ax12.text(0.02, y, line, fontsize=7.5, family='monospace',
                  transform=ax12.transAxes)
        y -= 0.075
    
    ax12.set_title('Group Means', fontsize=9, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Φ-Dwell Alzheimer's Eigenmode Analyzer")
    parser.add_argument('path', help='Path to ds004504 dataset')
    parser.add_argument('--use-derivatives', action='store_true',
                        help='Use preprocessed data from derivatives/')
    parser.add_argument('--max-duration', type=int, default=120,
                        help='Max seconds to analyze per subject (default: 120)')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════╗")
    print("║   Φ-DWELL ALZHEIMER'S EIGENMODE ANALYZER        ║")
    print("║   OpenNeuro ds004504                            ║")
    print("║   AD vs FTD vs Healthy Controls                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    
    # Build eigenmodes for 19-channel system
    graph_names, coords, eigenvecs, eigenvals = build_graph_laplacian(ELECTRODE_POS_19)
    print(f"Graph Laplacian: {len(graph_names)} electrodes, {N_MODES} modes")
    print(f"Eigenvalues: {eigenvals.round(2)}")
    print(f"Eigenmodes: {', '.join(MODE_NAMES)}")
    
    # Parse participant info
    participants = parse_participants(args.path)
    if participants:
        groups_count = Counter(p['group'] for p in participants.values())
        print(f"\nParticipants: {dict(groups_count)}")
        mmse_available = sum(1 for p in participants.values() if p['mmse'] is not None)
        print(f"MMSE scores available: {mmse_available}/{len(participants)}")
    else:
        print("\nWARNING: Could not parse participants.tsv")
    
    # Find EEG files
    file_map = find_eeg_files(args.path, args.use_derivatives)
    print(f"Found {len(file_map)} EEG files")
    
    if not file_map:
        print("\nERROR: No EEG files found. Check path structure.")
        print("Expected: path/sub-001/eeg/*.set or path/sub-001/eeg/*.edf")
        print("Or use --use-derivatives for preprocessed data")
        sys.exit(1)
    
    # Process each subject
    all_results = {}
    
    for subj_id, filepath in sorted(file_map.items()):
        pinfo = participants.get(subj_id, {})
        group = pinfo.get('group', '?')
        mmse = pinfo.get('mmse')
        group_label = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}.get(group, group)
        
        mmse_str = f"MMSE={mmse:.0f}" if mmse is not None else "MMSE=?"
        print(f"\n  {subj_id} [{group_label}] {mmse_str} — {os.path.basename(filepath)}")
        
        metrics = analyze_subject(filepath, graph_names, eigenvecs,
                                   max_duration_s=args.max_duration)
        
        if metrics is None:
            print(f"    FAILED")
            continue
        
        print(f"    {metrics['n_channels']}ch, {metrics['duration_s']:.0f}s, "
              f"{metrics['n_words']} words, vocab={metrics['vocab_size']}, "
              f"PP={metrics['perplexity']:.1f}, CV={metrics['mean_cv']:.2f}, "
              f"PSI={metrics.get('psi_mean', 0):.1f}±{metrics.get('psi_std', 0):.1f}")
        
        all_results[subj_id] = {
            'group': group,
            'mmse': mmse,
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'metrics': metrics,
        }
    
    if len(all_results) < 5:
        print("\nERROR: Too few subjects processed successfully")
        sys.exit(1)
    
    # ── Statistical Analysis ──
    print(f"\n\n{'═' * 65}")
    print(f"  STATISTICAL ANALYSIS ({len(all_results)} subjects)")
    print(f"{'═' * 65}")
    
    key_metrics = [
        ('vocab_size', 'Vocabulary Size'),
        ('entropy', 'Shannon Entropy'),
        ('perplexity', 'Bigram Perplexity'),
        ('self_rate', 'Self-Transition Rate'),
        ('mean_cv', 'Mean CV'),
        ('delta_theta_coupling', 'δ-θ Coupling'),
        ('mean_coupling', 'Mean Coupling'),
        ('top5_concentration', 'Top-5 Concentration'),
        ('criticality_fraction', 'Criticality Fraction'),
        ('zipf_alpha', 'Zipf α'),
        ('psi_mean', 'PSI Mean Iters'),
        ('psi_std', 'PSI Std'),
        ('psi_gradient', 'PSI Band Gradient'),
        ('psi_residual_mean', 'PSI Residual'),
        ('psi_fraction_converged', 'PSI % Converged'),
    ]
    
    print(f"\n  {'Metric':<24} {'CN':>10} {'AD':>10} {'FTD':>10} {'KW-p':>8} {'AD-CN p':>8} {'FTD-CN p':>8}")
    print(f"  {'─' * 80}")
    
    significant_metrics = []
    
    for metric, title in key_metrics:
        comp = group_comparison(all_results, metric)
        
        cn = comp.get('CN', {})
        ad = comp.get('AD', {})
        ftd = comp.get('FTD', {})
        
        cn_str = f"{cn.get('mean', 0):.3f}" if cn else "—"
        ad_str = f"{ad.get('mean', 0):.3f}" if ad else "—"
        ftd_str = f"{ftd.get('mean', 0):.3f}" if ftd else "—"
        
        kw_p = comp.get('kruskal_p', 1.0)
        ad_cn_p = comp.get('AD_vs_CN', {}).get('p', 1.0)
        ftd_cn_p = comp.get('FTD_vs_CN', {}).get('p', 1.0)
        
        sig = '***' if kw_p < 0.001 else ('**' if kw_p < 0.01 else ('*' if kw_p < 0.05 else ''))
        
        # Format large numbers differently
        if metric == 'vocab_size':
            cn_str = f"{cn.get('mean', 0):.0f}"
            ad_str = f"{ad.get('mean', 0):.0f}"
            ftd_str = f"{ftd.get('mean', 0):.0f}"
        elif metric == 'perplexity':
            cn_str = f"{cn.get('mean', 0):.1f}"
            ad_str = f"{ad.get('mean', 0):.1f}"
            ftd_str = f"{ftd.get('mean', 0):.1f}"
        
        print(f"  {title:<24} {cn_str:>10} {ad_str:>10} {ftd_str:>10} "
              f"{kw_p:>7.4f}{sig:>3} {ad_cn_p:>7.4f}  {ftd_cn_p:>7.4f}")
        
        if kw_p < 0.05:
            significant_metrics.append((metric, title, kw_p, comp))
    
    # MMSE correlations
    print(f"\n  MMSE CORRELATIONS")
    print(f"  {'─' * 60}")
    print(f"  {'Metric':<24} {'Spearman ρ':>12} {'p':>10}")
    print(f"  {'─' * 60}")
    
    for metric, title in key_metrics:
        mc = mmse_correlation(all_results, metric)
        rho = mc.get('spearman_rho', 0)
        p = mc.get('spearman_p', 1.0)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {title:<24} {rho:>11.3f}  {p:>9.4f} {sig}")
    
    # ── Summary ──
    print(f"\n  {'═' * 65}")
    print(f"  SUMMARY")
    print(f"  {'═' * 65}")
    print(f"  Significant group differences (KW p < 0.05): {len(significant_metrics)}/{len(key_metrics)}")
    for metric, title, p, comp in significant_metrics:
        ad_cn = comp.get('AD_vs_CN', {})
        ftd_cn = comp.get('FTD_vs_CN', {})
        ad_ftd = comp.get('AD_vs_FTD', {})
        print(f"    {title}: p={p:.4f}")
        if ad_cn.get('p', 1) < 0.05:
            print(f"      AD vs CN: p={ad_cn['p']:.4f}, effect={ad_cn['effect']:.3f}")
        if ftd_cn.get('p', 1) < 0.05:
            print(f"      FTD vs CN: p={ftd_cn['p']:.4f}, effect={ftd_cn['effect']:.3f}")
        if ad_ftd.get('p', 1) < 0.05:
            print(f"      AD vs FTD: p={ad_ftd['p']:.4f}, effect={ad_ftd['effect']:.3f}")
    
    # ── Plot ──
    fig_path = os.path.join(args.path, 'phidwell_alzheimer_results.png')
    plot_alzheimer_dashboard(all_results, fig_path)
    
    # ── Save JSON ──
    json_results = {}
    for subj_id, data in all_results.items():
        metrics = data['metrics'].copy()
        # Convert non-serializable items
        metrics.pop('coupling_matrix', None)
        json_results[subj_id] = {
            'group': {'A': 'AD', 'F': 'FTD', 'C': 'CN'}.get(data['group'], data['group']),
            'mmse': data['mmse'],
            'age': data.get('age'),
            **metrics,
        }
    
    json_path = os.path.join(args.path, 'phidwell_alzheimer_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")
    
    print(f"\n{'═' * 65}")
    print(f"  WHAT TO LOOK FOR")
    print(f"{'═' * 65}")
    print(f"""
  If Φ-Dwell detects Alzheimer's, you should see:
  
  1. VOCABULARY SIZE: CN > FTD > AD
     Alzheimer's destroys connectivity → fewer possible configurations
     
  2. PERPLEXITY: CN > FTD > AD  (or AD > CN if dynamics become random)
     More predictable = more constrained = less healthy
     OR: Higher perplexity in AD = dynamics become random/noisy
     The direction depends on disease stage.
     
  3. δ-θ COUPLING: CN > AD
     AD damages long-range tracts connecting frontal/temporal regions
     
  4. CRITICALITY (CV): CN > AD
     Healthy = critical (CV > 1). AD = shift toward random (CV → 1)
     
  5. SELF-TRANSITION: AD > CN (in alpha/theta bands specifically)
     AD dynamics more rigid, less flexible
     
  6. MMSE CORRELATION: Positive correlation with vocabulary/entropy
     More severe disease (lower MMSE) = smaller vocabulary
     
  7. PSI MEAN ITERATIONS: CN should show MODERATE convergence (5-10 iters)
     AD may show FAST convergence (collapsed to few eigenmodes = trivially
     recoverable) or SLOW (noisy, incoherent). The direction tells us
     whether AD dynamics are over-constrained or under-constrained.
     
  8. PSI GRADIENT: How convergence changes across frequency bands.
     Healthy brains should show a gradient (slow bands stable, fast bands
     dynamic). AD may flatten this gradient (loss of hierarchical structure).
     
  9. PSI FRACTION CONVERGED: What fraction of time windows achieve
     phase stability. Lower = more metastable dynamics. CN should be
     moderate; AD may be either too high (rigid) or too low (chaotic).
     
  FTD should show a DIFFERENT pattern from AD:
  - Specifically disrupted frontal modes (A-P eigenmode affected)
  - Posterior modes relatively preserved
  - PSI gradient may reverse (frontal instability > posterior)
  - This is the differential diagnosis signature
""")


if __name__ == '__main__':
    main()