#!/usr/bin/env python3
"""
Φ-Dwell Deep Analyzer — Extracting the Brain's Eigenmode Configuration Space
=============================================================================

Goes beyond the macroscope (which visualizes) to extract the scientific content:

1. EIGENMODE TRANSITION GRAMMAR
   - Which spatial mode follows which, per band
   - Semi-Markov transition probabilities with dwell-time-dependent transitions
   - "Words" in the brain's eigenmode language (stable sequences)

2. CROSS-BAND EIGENMODE COUPLING
   - When delta locks A-P, what does alpha do?
   - 40×40 coupling matrix: spatial-spectral interaction structure
   - A new kind of PAC: eigenmode-to-eigenmode across frequency bands

3. ATTRACTOR CATALOG
   - Identify stable attractor basins in the 40D state space
   - Name each by its eigenmode signature (dominant band × mode)
   - Dwell statistics, transition network between attractors
   - The brain's "computational repertoire"

4. MICRO-MACRO BRIDGE
   - Within-attractor dynamics (micro): dwell distributions, regime per attractor
   - Between-attractor dynamics (macro): transition graph, modularity
   - Scale-free structure: do attractor sizes follow a power law?

5. BRAIN CONFIGURATION FINGERPRINT
   - A compact summary that characterizes THIS brain in THIS state
   - Comparable across subjects, conditions, pathologies

Usage:
    python phidwell_deep_analyzer.py path/to/file.edf
    python phidwell_deep_analyzer.py path/to/eegmmidb/S001/ --band alpha
    python phidwell_deep_analyzer.py path/to/eegmmidb/ --subjects 20 --all-bands

Requirements:
    pip install numpy scipy matplotlib mne scikit-learn
"""

import numpy as np
import scipy.signal
import scipy.linalg
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from collections import Counter, defaultdict, deque
import argparse
import sys
import os
import glob
import time
import json
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
BAND_COLORS = ['#ff5050', '#ffb428', '#50ff50', '#50a0ff', '#c850ff']
MODE_NAMES = ['A-P', 'L-R', 'C-P', 'Diag', 'M5', 'M6', 'M7', 'M8']
N_MODES = 8

ELECTRODE_POS_64 = {
    'Fc5.': (-0.59, 0.31), 'Fc3.': (-0.35, 0.31), 'Fc1.': (-0.12, 0.31),
    'Fcz.': (0.0, 0.31), 'Fc2.': (0.12, 0.31), 'Fc4.': (0.35, 0.31),
    'Fc6.': (0.59, 0.31),
    'C5..': (-0.67, 0.0), 'C3..': (-0.40, 0.0), 'C1..': (-0.13, 0.0),
    'Cz..': (0.0, 0.0), 'C2..': (0.13, 0.0), 'C4..': (0.40, 0.0),
    'C6..': (0.67, 0.0),
    'Cp5.': (-0.59, -0.31), 'Cp3.': (-0.35, -0.31), 'Cp1.': (-0.12, -0.31),
    'Cpz.': (0.0, -0.31), 'Cp2.': (0.12, -0.31), 'Cp4.': (0.35, -0.31),
    'Cp6.': (0.59, -0.31),
    'Fp1.': (-0.25, 0.90), 'Fpz.': (0.0, 0.92), 'Fp2.': (0.25, 0.90),
    'Af7.': (-0.52, 0.75), 'Af3.': (-0.25, 0.72), 'Afz.': (0.0, 0.72),
    'Af4.': (0.25, 0.72), 'Af8.': (0.52, 0.75),
    'F7..': (-0.73, 0.52), 'F5..': (-0.55, 0.52), 'F3..': (-0.35, 0.52),
    'F1..': (-0.12, 0.52), 'Fz..': (0.0, 0.52), 'F2..': (0.12, 0.52),
    'F4..': (0.35, 0.52), 'F6..': (0.55, 0.52), 'F8..': (0.73, 0.52),
    'Ft7.': (-0.80, 0.18), 'Ft8.': (0.80, 0.18),
    'T7..': (-0.85, 0.0), 'T8..': (0.85, 0.0),
    'T9..': (-0.95, 0.0), 'T10.': (0.95, 0.0),
    'Tp7.': (-0.80, -0.18), 'Tp8.': (0.80, -0.18),
    'P7..': (-0.73, -0.52), 'P5..': (-0.55, -0.52), 'P3..': (-0.35, -0.52),
    'P1..': (-0.12, -0.52), 'Pz..': (0.0, -0.52), 'P2..': (0.12, -0.52),
    'P4..': (0.35, -0.52), 'P6..': (0.55, -0.52), 'P8..': (0.73, -0.52),
    'Po7.': (-0.45, -0.72), 'Po3.': (-0.25, -0.72), 'Poz.': (0.0, -0.72),
    'Po4.': (0.25, -0.72), 'Po8.': (0.45, -0.72),
    'O1..': (-0.25, -0.87), 'Oz..': (0.0, -0.87), 'O2..': (0.25, -0.87),
    'Iz..': (0.0, -0.95),
}

ELECTRODE_POS_20 = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6),
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0),
    'C4': (0.4, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5),
    'P4': (0.35, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85),
}


# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN
# ═══════════════════════════════════════════════════════════════

def build_graph(positions, sigma=0.5):
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            A[i, j] = np.exp(-d**2 / (2 * sigma**2))
            A[j, i] = A[i, j]
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = scipy.linalg.eigh(L)
    return names, coords, eigvals, eigvecs


def map_channels(raw_ch_names, electrode_names):
    mapping = {}
    clean_lookup = {e.replace('.', '').lower(): e for e in electrode_names}
    for ch in raw_ch_names:
        clean = (ch.replace('EEG ', '').replace('EEG', '')
                 .replace('-REF', '').replace('-Ref', '').replace('-ref', '')
                 .strip().replace(' ', '').replace('.', '').lower()
                 .replace('t3', 't7').replace('t4', 't8')
                 .replace('t5', 'p7').replace('t6', 'p8'))
        if clean in clean_lookup:
            mapping[ch] = clean_lookup[clean]
        else:
            for ename in electrode_names:
                eclean = ename.replace('.', '').lower()
                if clean == eclean or clean.rstrip('.') == eclean:
                    mapping[ch] = ename
                    break
    return mapping


# ═══════════════════════════════════════════════════════════════
# CORE: Extract 40D eigenmode timeseries from EEG
# ═══════════════════════════════════════════════════════════════

def extract_eigenmode_timeseries(edf_path, window_ms=250, step_ms=10):
    """
    The foundational extraction: load EEG, compute graph Laplacian eigenmodes,
    project phase fields onto eigenmodes at every time step, for all 5 bands.
    
    Returns:
        times: (T,) time in seconds
        coeffs: (T, 5, 8) eigenmode coefficient magnitudes
        phases: (T, 5, 8) eigenmode phases
        meta: dict with sfreq, n_channels, eigvals, etc.
    """
    print(f"  Loading: {os.path.basename(edf_path)}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samples = data.shape[1]

    # Detect electrode system
    n_eeg = sum(1 for ch in raw.ch_names
                if not ch.startswith('STI') and 'annotation' not in ch.lower())
    positions = ELECTRODE_POS_64 if n_eeg >= 60 else ELECTRODE_POS_20
    
    e_names, e_coords, eigvals, eigvecs = build_graph(positions)
    mapping = map_channels(raw.ch_names, e_names)
    print(f"    {len(mapping)}/{len(e_names)} channels mapped, sfreq={sfreq}Hz, "
          f"dur={n_samples/sfreq:.1f}s")

    # Eigenvectors
    n_modes = N_MODES
    V = eigvecs[:, 1:n_modes + 1].copy()

    # Pre-compute band-filtered phases
    all_band_phases = {}
    active_mask = np.zeros(len(e_names), dtype=bool)

    for band_name, (lo, hi) in BANDS.items():
        nyq = sfreq / 2.0
        lo_n = max(0.5, lo) / nyq
        hi_n = min(hi, nyq - 1) / nyq
        b, a = scipy.signal.butter(3, [lo_n, hi_n], btype='band')

        phases = np.zeros((len(e_names), n_samples), dtype=np.float32)
        for ch_raw, ch_graph in mapping.items():
            idx_g = e_names.index(ch_graph)
            idx_r = raw.ch_names.index(ch_raw)
            filtered = scipy.signal.filtfilt(b, a, data[idx_r])
            analytic = scipy.signal.hilbert(filtered)
            phases[idx_g] = np.angle(analytic).astype(np.float32)
            active_mask[idx_g] = True

        all_band_phases[band_name] = phases

    # Mask and normalize eigenvectors
    V_masked = V.copy()
    V_masked[~active_mask] = 0
    for m in range(n_modes):
        norm = np.linalg.norm(V_masked[:, m])
        if norm > 1e-10:
            V_masked[:, m] /= norm

    # Sliding window extraction
    window_samp = int(window_ms / 1000.0 * sfreq)
    step_samp = max(1, int(step_ms / 1000.0 * sfreq))
    n_windows = (n_samples - window_samp) // step_samp

    times = np.zeros(n_windows)
    coeffs = np.zeros((n_windows, 5, n_modes))
    phases_out = np.zeros((n_windows, 5, n_modes))

    t0 = time.time()
    for i in range(n_windows):
        start = i * step_samp
        center = start + window_samp // 2
        times[i] = center / sfreq

        for bi, band in enumerate(BAND_NAMES):
            bp = all_band_phases[band]
            sin_m = np.mean(np.sin(bp[:, start:start + window_samp]), axis=1)
            cos_m = np.mean(np.cos(bp[:, start:start + window_samp]), axis=1)
            mean_phase = np.arctan2(sin_m, cos_m)

            z_r = np.cos(mean_phase)
            z_i = np.sin(mean_phase)

            for m in range(n_modes):
                pr = np.dot(z_r, V_masked[:, m])
                pi = np.dot(z_i, V_masked[:, m])
                coeffs[i, bi, m] = np.sqrt(pr**2 + pi**2)
                phases_out[i, bi, m] = np.arctan2(pi, pr)

        if (i + 1) % (n_windows // 5) == 0:
            print(f"    {100*(i+1)//n_windows}% ({(i+1)/(time.time()-t0):.0f} win/s)")

    print(f"    {n_windows} windows in {time.time()-t0:.1f}s")

    meta = {
        'sfreq': sfreq,
        'n_channels': len(mapping),
        'n_windows': n_windows,
        'window_ms': window_ms,
        'step_ms': step_ms,
        'eigvals': eigvals[1:n_modes+1],
        'e_names': e_names,
        'e_coords': e_coords,
        'eigvecs': eigvecs,
        'system': '64ch' if n_eeg >= 60 else '20ch',
    }
    return times, coeffs, phases_out, meta


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: Eigenmode Transition Grammar
# ═══════════════════════════════════════════════════════════════

def extract_transition_grammar(phases, step_ms=10, threshold_deg=45):
    """
    Extract the transition grammar of eigenmode dominance sequences.
    
    For each band, identify the dominant eigenmode at each timepoint,
    compute the transition matrix, and find recurring "words" (stable sequences).
    """
    threshold = np.radians(threshold_deg)
    T, n_bands, n_modes = phases.shape
    
    results = {}
    
    for bi, band in enumerate(BAND_NAMES):
        # Detect dwells and transitions per mode
        phase_traj = phases[:, bi, :]  # (T, n_modes)
        
        # Dominant mode at each timepoint
        # Use coefficient magnitude from the caller, but here we use
        # phase stability: which mode's phase changes least
        diffs = np.abs(np.diff(phase_traj, axis=0))
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        
        # Most stable mode at each step
        stability = 1.0 / (np.mean(diffs, axis=0) + 0.01)  # inverse of mean phase velocity
        
        # Dominant mode sequence (mode with highest coefficient at each time)
        # We'll use a simpler approach: mode with lowest phase velocity in a small window
        win = 5  # 50ms smoothing
        dom_modes = np.zeros(T - 1, dtype=int)
        for t in range(T - 1):
            t0 = max(0, t - win)
            t1 = min(T - 1, t + win)
            local_stability = np.mean(diffs[t0:t1], axis=0)
            dom_modes[t] = np.argmin(local_stability)
        
        # Transition matrix
        trans_matrix = np.zeros((n_modes, n_modes))
        for t in range(len(dom_modes) - 1):
            trans_matrix[dom_modes[t], dom_modes[t + 1]] += 1
        
        # Normalize
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_probs = np.divide(trans_matrix, row_sums, where=row_sums > 0,
                                out=np.zeros_like(trans_matrix))
        
        # Self-transition rate (dwell probability)
        self_trans = np.diag(trans_probs)
        
        # Find "words": sequences of same dominant mode
        words = []
        current_mode = dom_modes[0]
        current_len = 1
        for t in range(1, len(dom_modes)):
            if dom_modes[t] == current_mode:
                current_len += 1
            else:
                words.append((current_mode, current_len * step_ms))
                current_mode = dom_modes[t]
                current_len = 1
        words.append((current_mode, current_len * step_ms))
        
        # Word statistics
        word_durations = defaultdict(list)
        for mode, dur in words:
            word_durations[mode].append(dur)
        
        word_stats = {}
        for mode in range(n_modes):
            durs = word_durations.get(mode, [step_ms])
            word_stats[MODE_NAMES[mode]] = {
                'count': len(durs),
                'mean_ms': float(np.mean(durs)),
                'max_ms': float(np.max(durs)),
                'cv': float(np.std(durs) / (np.mean(durs) + 1e-9)),
            }
        
        # Bigrams: which mode pairs occur most
        bigrams = Counter()
        for t in range(len(dom_modes) - 1):
            if dom_modes[t] != dom_modes[t + 1]:
                bigrams[(MODE_NAMES[dom_modes[t]], MODE_NAMES[dom_modes[t+1]])] += 1
        
        results[band] = {
            'trans_matrix': trans_matrix,
            'trans_probs': trans_probs,
            'self_trans': self_trans,
            'words': words,
            'word_stats': word_stats,
            'top_bigrams': bigrams.most_common(10),
            'dom_mode_sequence': dom_modes,
        }
    
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: Cross-Band Eigenmode Coupling
# ═══════════════════════════════════════════════════════════════

def compute_cross_band_coupling(coeffs):
    """
    Compute the 40×40 cross-band eigenmode coupling matrix.
    
    Element (i, j) tells you: when eigenmode i in band X is strong,
    is eigenmode j in band Y also strong?
    
    This is a new kind of PAC — spatial-spectral coupling.
    """
    T, n_bands, n_modes = coeffs.shape
    D = n_bands * n_modes  # 40
    
    # Flatten to (T, 40)
    flat = coeffs.reshape(T, D)
    
    # Remove zero-variance columns (would cause NaN in correlation)
    col_std = np.std(flat, axis=0)
    valid = col_std > 1e-10
    if np.sum(valid) < 2:
        # Not enough variance for correlation
        coupling = np.zeros((D, D))
    else:
        # Add tiny noise to prevent exact-zero variance
        flat_safe = flat + np.random.randn(*flat.shape) * 1e-12
        coupling = np.corrcoef(flat_safe.T)  # (40, 40)
        coupling = np.nan_to_num(coupling, nan=0.0)
    
    # Block structure: extract between-band coupling
    between_band = np.zeros((n_bands, n_bands))
    for bi in range(n_bands):
        for bj in range(n_bands):
            block = coupling[bi*n_modes:(bi+1)*n_modes,
                             bj*n_modes:(bj+1)*n_modes]
            # Mean absolute off-diagonal coupling
            if bi == bj:
                mask = ~np.eye(n_modes, dtype=bool)
                between_band[bi, bj] = np.mean(np.abs(block[mask]))
            else:
                between_band[bi, bj] = np.mean(np.abs(block))
    
    # Strongest cross-band pairs
    pairs = []
    for bi in range(n_bands):
        for bj in range(bi + 1, n_bands):
            for mi in range(n_modes):
                for mj in range(n_modes):
                    r = coupling[bi * n_modes + mi, bj * n_modes + mj]
                    pairs.append({
                        'band_i': BAND_NAMES[bi], 'mode_i': MODE_NAMES[mi],
                        'band_j': BAND_NAMES[bj], 'mode_j': MODE_NAMES[mj],
                        'r': float(r),
                    })
    
    pairs.sort(key=lambda x: abs(x['r']), reverse=True)
    
    return {
        'full_matrix': coupling,
        'between_band': between_band,
        'top_pairs': pairs[:20],
        'mean_coupling': float(np.mean(np.abs(coupling))),
    }


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: Attractor Catalog
# ═══════════════════════════════════════════════════════════════

def build_attractor_catalog(coeffs, phases, times, step_ms=10,
                            n_attractors=12, min_dwell_ms=50):
    """
    Identify stable attractor basins in the 40D eigenmode state space.
    
    Each attractor is characterized by:
    - Its eigenmode signature (which modes/bands are dominant)
    - Dwell time distribution within the attractor
    - Transition probabilities to other attractors
    - Dynamical regime (critical/bursty/clocklike)
    """
    T, n_bands, n_modes = coeffs.shape
    D = n_bands * n_modes
    flat = coeffs.reshape(T, D)
    
    # Normalize for clustering
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std[std < 1e-9] = 1
    normalized = (flat - mean) / std
    
    # K-means clustering
    n_clust = min(n_attractors, T // 20)
    km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
    labels = km.fit_predict(normalized)
    centers = km.cluster_centers_  # (n_clust, 40)
    
    # Decode centers back to (n_bands, n_modes)
    centers_decoded = (centers * std + mean).reshape(n_clust, n_bands, n_modes)
    
    # Build attractor catalog
    catalog = []
    for ci in range(n_clust):
        mask = labels == ci
        n_frames = np.sum(mask)
        
        # Dominant band and mode
        center = centers_decoded[ci]
        band_powers = np.sum(center, axis=1)
        dom_band = int(np.argmax(band_powers))
        dom_mode = int(np.argmax(center[dom_band]))
        
        # Dwell times within this attractor
        runs = []
        current_run = 0
        for t in range(T):
            if labels[t] == ci:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run * step_ms)
                current_run = 0
        if current_run > 0:
            runs.append(current_run * step_ms)
        
        runs = np.array(runs) if runs else np.array([step_ms])
        
        # Regime within attractor
        mean_d = np.mean(runs)
        cv = np.std(runs) / (mean_d + 1e-9)
        kurt = float(stats.kurtosis(runs)) if len(runs) > 4 else 0
        
        if cv < 0.7:
            regime = 'clocklike'
        elif cv > 1.3 and kurt > 50:
            regime = 'bursty'
        elif cv > 1.1:
            regime = 'critical'
        else:
            regime = 'random'
        
        # Eigenmode signature: normalized coefficient vector
        signature = center / (np.max(center) + 1e-9)
        
        # Name the attractor
        name = f"{BAND_NAMES[dom_band].upper()}-{MODE_NAMES[dom_mode]}"
        
        catalog.append({
            'id': ci,
            'name': name,
            'n_frames': int(n_frames),
            'fraction': float(n_frames / T),
            'dom_band': dom_band,
            'dom_band_name': BAND_NAMES[dom_band],
            'dom_mode': dom_mode,
            'dom_mode_name': MODE_NAMES[dom_mode],
            'mean_dwell_ms': float(np.mean(runs)),
            'max_dwell_ms': float(np.max(runs)),
            'n_visits': len(runs),
            'cv': float(cv),
            'kurtosis': float(kurt),
            'regime': regime,
            'signature': signature,
            'band_powers': band_powers,
        })
    
    # Sort by fraction (time spent)
    catalog.sort(key=lambda x: x['fraction'], reverse=True)
    
    # Transition matrix between attractors
    trans = np.zeros((n_clust, n_clust))
    for t in range(T - 1):
        if labels[t] != labels[t + 1]:
            trans[labels[t], labels[t + 1]] += 1
    
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_probs = np.divide(trans, row_sums, where=row_sums > 0,
                            out=np.zeros_like(trans))
    
    # Attractor graph metrics
    from scipy.sparse.csgraph import connected_components
    n_comp, _ = connected_components(trans > 0)
    
    # Modularity approximation
    total_trans = np.sum(trans)
    if total_trans > 0:
        ki = trans.sum(axis=1)
        kj = trans.sum(axis=0)
        expected = np.outer(ki, kj) / total_trans
        Q = np.sum((trans - expected) * (labels[:, None] == labels[None, :]).astype(float) 
                    if False else 0)  # Simplified
    
    return {
        'catalog': catalog,
        'labels': labels,
        'trans_matrix': trans,
        'trans_probs': trans_probs,
        'n_attractors': n_clust,
        'n_components': n_comp,
        'flat_trajectory': flat,
    }


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Brain Configuration Fingerprint
# ═══════════════════════════════════════════════════════════════

def compute_fingerprint(grammar, coupling, attractors, meta):
    """
    Compact fingerprint that characterizes this brain in this state.
    A vector of ~50 numbers that can be compared across subjects.
    """
    fp = {}
    
    # 1. Band-wise grammar statistics (5 bands × 3 metrics = 15)
    for band in BAND_NAMES:
        g = grammar[band]
        fp[f'{band}_self_trans_mean'] = float(np.mean(g['self_trans']))
        fp[f'{band}_self_trans_max'] = float(np.max(g['self_trans']))
        fp[f'{band}_n_words'] = sum(s['count'] for s in g['word_stats'].values())
    
    # 2. Cross-band coupling summary (10 pairs = 10)
    cb = coupling['between_band']
    for bi in range(5):
        for bj in range(bi + 1, 5):
            fp[f'coupling_{BAND_NAMES[bi]}_{BAND_NAMES[bj]}'] = float(cb[bi, bj])
    
    # 3. Attractor catalog summary (5)
    cat = attractors['catalog']
    fp['n_attractors_used'] = len([a for a in cat if a['fraction'] > 0.05])
    fp['top_attractor_fraction'] = cat[0]['fraction'] if cat else 0
    fp['attractor_entropy'] = float(stats.entropy([a['fraction'] for a in cat]))
    
    regimes = Counter(a['regime'] for a in cat)
    fp['fraction_critical'] = regimes.get('critical', 0) / len(cat) if cat else 0
    fp['fraction_clocklike'] = regimes.get('clocklike', 0) / len(cat) if cat else 0
    
    # 4. Dominant attractor signature (8)
    if cat:
        sig = cat[0]['signature']
        for bi in range(min(5, sig.shape[0])):
            fp[f'top_sig_band{bi}'] = float(np.max(sig[bi]))
    
    # 5. Overall metastability index (1)
    cvs = [a['cv'] for a in cat]
    fp['global_metastability'] = float(np.mean(cvs)) if cvs else 0
    
    return fp


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def create_deep_analysis_figure(grammar, coupling, attractors, fingerprint, 
                                 meta, output_path):
    """Generate the comprehensive analysis figure."""
    
    fig = plt.figure(figsize=(24, 18), facecolor='#0a0a12')
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4,
                  left=0.05, right=0.97, top=0.93, bottom=0.04)
    
    fig.suptitle(
        f'Φ-DWELL DEEP ANALYZER   ·   {meta["system"]}   ·   '
        f'{meta["n_channels"]}ch   ·   {meta["n_windows"]} windows',
        fontsize=14, fontweight='bold', color='#b44adc',
        fontfamily='monospace', y=0.97
    )
    
    # ── Panel 1: Transition Grammar (alpha band) ──
    ax = fig.add_subplot(gs[0, 0])
    band = 'alpha'
    tp = grammar[band]['trans_probs']
    im = ax.imshow(tp, cmap='inferno', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(N_MODES))
    ax.set_yticks(range(N_MODES))
    ax.set_xticklabels(MODE_NAMES, fontsize=6, color='#aaaacc')
    ax.set_yticklabels(MODE_NAMES, fontsize=6, color='#aaaacc')
    ax.set_title(f'TRANSITION GRAMMAR (α)', fontsize=9, color='#8888aa',
                  fontfamily='monospace')
    ax.set_xlabel('To', fontsize=7, color='#5a5a7a')
    ax.set_ylabel('From', fontsize=7, color='#5a5a7a')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 2: Transition Grammar (theta band) ──
    ax = fig.add_subplot(gs[0, 1])
    tp = grammar['theta']['trans_probs']
    im = ax.imshow(tp, cmap='inferno', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(N_MODES))
    ax.set_yticks(range(N_MODES))
    ax.set_xticklabels(MODE_NAMES, fontsize=6, color='#aaaacc')
    ax.set_yticklabels(MODE_NAMES, fontsize=6, color='#aaaacc')
    ax.set_title(f'TRANSITION GRAMMAR (θ)', fontsize=9, color='#8888aa',
                  fontfamily='monospace')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 3: Cross-Band Coupling Matrix ──
    ax = fig.add_subplot(gs[0, 2])
    cb = coupling['between_band']
    im = ax.imshow(cb, cmap='magma', vmin=0, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'], fontsize=9, color='#aaaacc')
    ax.set_yticklabels(['δ', 'θ', 'α', 'β', 'γ'], fontsize=9, color='#aaaacc')
    ax.set_title('CROSS-BAND COUPLING', fontsize=9, color='#8888aa',
                  fontfamily='monospace')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 4: Full 40×40 Coupling Matrix ──
    ax = fig.add_subplot(gs[0, 3])
    fm = coupling['full_matrix']
    im = ax.imshow(fm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    # Draw band boundaries
    for b in range(1, 5):
        ax.axhline(b * N_MODES - 0.5, color='white', linewidth=0.3, alpha=0.5)
        ax.axvline(b * N_MODES - 0.5, color='white', linewidth=0.3, alpha=0.5)
    ax.set_title('40D COUPLING (all bands × modes)', fontsize=8, color='#8888aa',
                  fontfamily='monospace')
    ax.set_xlabel('δ    θ    α    β    γ', fontsize=7, color='#5a5a7a')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 5: Attractor Catalog (bar chart) ──
    ax = fig.add_subplot(gs[1, 0:2])
    cat = attractors['catalog']
    n_show = min(12, len(cat))
    names = [a['name'] for a in cat[:n_show]]
    fracs = [a['fraction'] for a in cat[:n_show]]
    colors = [BAND_COLORS[a['dom_band']] for a in cat[:n_show]]
    
    bars = ax.barh(range(n_show), fracs, color=colors, alpha=0.8)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(names, fontsize=8, color='#aaaacc', fontfamily='monospace')
    ax.invert_yaxis()
    ax.set_xlabel('Fraction of time', fontsize=8, color='#5a5a7a')
    ax.set_title('ATTRACTOR CATALOG', fontsize=9, color='#8888aa',
                  fontfamily='monospace')
    
    # Add regime labels
    for i, a in enumerate(cat[:n_show]):
        ax.text(fracs[i] + 0.005, i, 
                f" {a['regime'][:4]}  μ={a['mean_dwell_ms']:.0f}ms  CV={a['cv']:.1f}",
                fontsize=7, color='#8888aa', va='center', fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 6: Attractor Transition Network ──
    ax = fig.add_subplot(gs[1, 2:4])
    tp = attractors['trans_probs']
    n_att = len(cat)
    
    # PCA of attractor centers for layout
    if n_att >= 3:
        sigs = np.array([a['signature'].flatten() for a in cat])
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(sigs)
    else:
        pos_2d = np.random.randn(n_att, 2)
    
    # Draw edges
    for i in range(n_att):
        for j in range(n_att):
            if i != j and tp[cat[i]['id'], cat[j]['id']] > 0.05:
                w = tp[cat[i]['id'], cat[j]['id']]
                ax.annotate('', xy=pos_2d[j], xytext=pos_2d[i],
                           arrowprops=dict(arrowstyle='->', color='#4a4a6a',
                                          lw=w * 3, alpha=0.5))
    
    # Draw nodes
    for i, a in enumerate(cat):
        c = BAND_COLORS[a['dom_band']]
        size = a['fraction'] * 800 + 50
        ax.scatter(pos_2d[i, 0], pos_2d[i, 1], s=size, c=c,
                  alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
        ax.text(pos_2d[i, 0], pos_2d[i, 1] - 0.15, a['name'],
               fontsize=6, color='white', ha='center', fontfamily='monospace')
    
    ax.set_title('ATTRACTOR NETWORK', fontsize=9, color='#8888aa',
                  fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ── Panel 7: Phase portrait (PCA of 40D trajectory) ──
    ax = fig.add_subplot(gs[2, 0:2])
    flat = attractors['flat_trajectory']
    pca = PCA(n_components=2)
    proj = pca.fit_transform(flat)
    
    # Color by dominant band at each timepoint
    bp = np.sum(flat.reshape(-1, 5, N_MODES), axis=2)  # (T, 5)
    dom = np.argmax(bp, axis=1)
    
    for bi in range(5):
        mask = dom == bi
        ax.scatter(proj[mask, 0], proj[mask, 1], c=BAND_COLORS[bi],
                  s=1, alpha=0.15, label=BAND_NAMES[bi])
    
    # Trail for last 200 points
    n = min(200, len(proj))
    ax.plot(proj[-n:, 0], proj[-n:, 1], color='white', linewidth=0.5, alpha=0.4)
    ax.scatter(proj[-1, 0], proj[-1, 1], c='white', s=40, zorder=10, marker='*')
    
    ax.set_title(f'40D PHASE PORTRAIT (PCA, {pca.explained_variance_ratio_[:2].sum()*100:.0f}% var)',
                 fontsize=9, color='#8888aa', fontfamily='monospace')
    ax.legend(fontsize=7, markerscale=5, framealpha=0.3)
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ── Panel 8: Self-transition rates across bands ──
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(N_MODES)
    for bi, band in enumerate(BAND_NAMES):
        st = grammar[band]['self_trans']
        ax.plot(x, st, 'o-', color=BAND_COLORS[bi], markersize=4,
               label=band, alpha=0.8, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODE_NAMES, fontsize=6, color='#aaaacc')
    ax.set_ylabel('Self-transition probability', fontsize=7, color='#5a5a7a')
    ax.set_title('MODE PERSISTENCE (per band)', fontsize=8, color='#8888aa',
                  fontfamily='monospace')
    ax.legend(fontsize=6, framealpha=0.3)
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ── Panel 9: Top cross-band coupling pairs ──
    ax = fig.add_subplot(gs[2, 3])
    top = coupling['top_pairs'][:15]
    labels_c = [f"{p['band_i'][0]}{p['mode_i']}-{p['band_j'][0]}{p['mode_j']}" for p in top]
    vals = [p['r'] for p in top]
    colors_c = ['#50ffb4' if v > 0 else '#ff5050' for v in vals]
    ax.barh(range(len(top)), vals, color=colors_c, alpha=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels_c, fontsize=6, color='#aaaacc', fontfamily='monospace')
    ax.invert_yaxis()
    ax.axvline(0, color='white', linewidth=0.5)
    ax.set_title('TOP CROSS-BAND PAIRS', fontsize=8, color='#8888aa',
                  fontfamily='monospace')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ── Panel 10: Fingerprint radar chart ──
    ax = fig.add_subplot(gs[3, 0], polar=True)
    fp_keys = ['global_metastability', 'attractor_entropy', 'fraction_critical',
               'top_attractor_fraction', 'n_attractors_used']
    fp_vals = [fingerprint.get(k, 0) for k in fp_keys]
    # Normalize to 0-1
    fp_max = [1.5, 3.0, 1.0, 0.5, 12]
    fp_norm = [min(1, v / m) for v, m in zip(fp_vals, fp_max)]
    fp_norm.append(fp_norm[0])  # close the polygon
    
    angles = np.linspace(0, 2 * np.pi, len(fp_keys), endpoint=False).tolist()
    angles.append(angles[0])
    
    ax.fill(angles, fp_norm, color='#b44adc', alpha=0.2)
    ax.plot(angles, fp_norm, 'o-', color='#b44adc', linewidth=1.5, markersize=4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Meta', 'Entropy', 'Critical', 'TopAttr', '#Attr'],
                        fontsize=7, color='#8888aa')
    ax.set_title('BRAIN FINGERPRINT', fontsize=8, color='#8888aa',
                  fontfamily='monospace', pad=15)
    ax.set_facecolor('#0f0f1a')
    
    # ── Panel 11: Word length distributions (all bands) ──
    ax = fig.add_subplot(gs[3, 1:3])
    for bi, band in enumerate(BAND_NAMES):
        words = grammar[band]['words']
        durs = [w[1] for w in words]
        if len(durs) > 10:
            sorted_d = np.sort(durs)[::-1]
            ccdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.loglog(sorted_d, ccdf, '.', color=BAND_COLORS[bi], markersize=3,
                     alpha=0.7, label=band)
    
    ax.set_xlabel('Word duration (ms)', fontsize=7, color='#5a5a7a')
    ax.set_ylabel('P(X > x)', fontsize=7, color='#5a5a7a')
    ax.set_title('EIGENMODE WORD SURVIVAL (log-log)', fontsize=8, color='#8888aa',
                  fontfamily='monospace')
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.1, which='both')
    ax.set_facecolor('#0f0f1a')
    ax.tick_params(colors='#3a3a5a', labelsize=6)
    
    # ── Panel 12: Fingerprint values table ──
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    ax.set_facecolor('#0f0f1a')
    
    # Print key fingerprint values
    y = 0.95
    ax.text(0.05, y, 'CONFIGURATION FINGERPRINT', fontsize=8, color='#b44adc',
            fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
    y -= 0.08
    for key, val in sorted(fingerprint.items())[:15]:
        short_key = key.replace('_', ' ')[:25]
        ax.text(0.05, y, f'{short_key}', fontsize=6, color='#5a5a7a',
                fontfamily='monospace', transform=ax.transAxes)
        ax.text(0.75, y, f'{val:.3f}', fontsize=7, color='#c8c8e8',
                fontfamily='monospace', transform=ax.transAxes)
        y -= 0.06
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a12')
    print(f"\n  Saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def analyze_file(edf_path, output_dir=None):
    """Run the complete deep analysis on one EDF file."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(edf_path))
    
    # 1. Extract eigenmode timeseries
    print("\n═══ EXTRACTING EIGENMODE TIMESERIES ═══")
    times, coeffs, phases, meta = extract_eigenmode_timeseries(edf_path)
    
    # 2. Transition grammar
    print("\n═══ EIGENMODE TRANSITION GRAMMAR ═══")
    grammar = extract_transition_grammar(phases, step_ms=meta['step_ms'])
    
    for band in BAND_NAMES:
        g = grammar[band]
        print(f"\n  {band.upper()}:")
        print(f"    Self-transition rates: {', '.join(f'{MODE_NAMES[m]}={g['self_trans'][m]:.2f}' for m in range(N_MODES))}")
        print(f"    Top bigrams: {g['top_bigrams'][:5]}")
        ws = g['word_stats']
        longest = max(ws.items(), key=lambda x: x[1]['mean_ms'])
        print(f"    Longest-dwelling mode: {longest[0]} ({longest[1]['mean_ms']:.0f}ms mean, "
              f"CV={longest[1]['cv']:.2f})")
    
    # 3. Cross-band coupling
    print("\n═══ CROSS-BAND EIGENMODE COUPLING ═══")
    coupling = compute_cross_band_coupling(coeffs)
    
    print(f"  Mean coupling strength: {coupling['mean_coupling']:.3f}")
    print(f"\n  Between-band coupling matrix:")
    cb = coupling['between_band']
    print(f"         {'  '.join(f'{b[:3]:>5}' for b in BAND_NAMES)}")
    for bi in range(5):
        row = '  '.join(f'{cb[bi, bj]:>5.3f}' for bj in range(5))
        print(f"  {BAND_NAMES[bi][:5]:>5} {row}")
    
    print(f"\n  Top 5 cross-band pairs:")
    for p in coupling['top_pairs'][:5]:
        print(f"    {p['band_i']}-{p['mode_i']} ↔ {p['band_j']}-{p['mode_j']}: "
              f"r={p['r']:+.3f}")
    
    # 4. Attractor catalog
    print("\n═══ ATTRACTOR CATALOG ═══")
    attractors = build_attractor_catalog(coeffs, phases, times, 
                                          step_ms=meta['step_ms'])
    
    print(f"\n  {'#':>3} {'Name':>12} {'Frac':>6} {'Dwell':>8} {'Visits':>7} "
          f"{'CV':>5} {'Regime':>10}")
    print("  " + "─" * 60)
    for a in attractors['catalog'][:12]:
        print(f"  {a['id']:>3} {a['name']:>12} {a['fraction']:>5.1%} "
              f"{a['mean_dwell_ms']:>6.0f}ms {a['n_visits']:>6} "
              f"{a['cv']:>5.2f} {a['regime']:>10}")
    
    # 5. Fingerprint
    print("\n═══ BRAIN CONFIGURATION FINGERPRINT ═══")
    fingerprint = compute_fingerprint(grammar, coupling, attractors, meta)
    
    for key, val in sorted(fingerprint.items()):
        print(f"  {key:>35}: {val:.4f}")
    
    # 6. Generate figure
    print("\n═══ GENERATING FIGURE ═══")
    basename = os.path.splitext(os.path.basename(edf_path))[0]
    output_path = os.path.join(output_dir, f'phidwell_deep_{basename}.png')
    create_deep_analysis_figure(grammar, coupling, attractors, fingerprint,
                                 meta, output_path)
    
    # 7. Save fingerprint as JSON
    fp_path = os.path.join(output_dir, f'phidwell_fingerprint_{basename}.json')
    with open(fp_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    print(f"  Saved fingerprint: {fp_path}")
    
    return grammar, coupling, attractors, fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Φ-Dwell Deep Analyzer — Brain eigenmode configuration space extraction"
    )
    parser.add_argument('data_path',
                        help='Path to EDF file or eegmmidb subject directory')
    parser.add_argument('--subjects', type=int, default=1,
                        help='Number of subjects for multi-subject analysis')
    parser.add_argument('--output', default=None,
                        help='Output directory')

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   Φ-DWELL DEEP ANALYZER                         ║")
    print("║   Brain Eigenmode Configuration Space            ║")
    print("║   Grammar · Coupling · Attractors · Fingerprint  ║")
    print("╚══════════════════════════════════════════════════╝")

    path = args.data_path
    output_dir = args.output

    if os.path.isfile(path) and path.lower().endswith('.edf'):
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(path))
        analyze_file(path, output_dir)
    elif os.path.isdir(path):
        # Find EDF files
        edfs = sorted(glob.glob(os.path.join(path, '*R01*.edf')))
        if not edfs:
            edfs = sorted(glob.glob(os.path.join(path, '*.edf')))
        
        if args.subjects > 1:
            # Multi-subject: look for subject directories
            sdirs = sorted(glob.glob(os.path.join(path, 'S[0-9][0-9][0-9]')))
            if not sdirs:
                sdirs = sorted(glob.glob(os.path.join(path, '1.0.0', 'S[0-9][0-9][0-9]')))
            
            sdirs = sdirs[:args.subjects]
            if output_dir is None:
                output_dir = path
            
            all_fingerprints = {}
            for si, sdir in enumerate(sdirs):
                sname = os.path.basename(sdir)
                print(f"\n{'='*60}")
                print(f"Subject {sname} ({si+1}/{len(sdirs)})")
                print(f"{'='*60}")
                
                s_edfs = sorted(glob.glob(os.path.join(sdir, '*R01*.edf')))
                if not s_edfs:
                    s_edfs = sorted(glob.glob(os.path.join(sdir, '*.edf')))
                if s_edfs:
                    try:
                        _, _, _, fp = analyze_file(s_edfs[0], output_dir)
                        all_fingerprints[sname] = fp
                    except Exception as e:
                        print(f"  FAILED: {e}")
            
            # Save all fingerprints
            if all_fingerprints:
                fp_all_path = os.path.join(output_dir, 'phidwell_all_fingerprints.json')
                with open(fp_all_path, 'w') as f:
                    json.dump(all_fingerprints, f, indent=2)
                print(f"\nAll fingerprints saved: {fp_all_path}")
        else:
            if edfs:
                if output_dir is None:
                    output_dir = path
                analyze_file(edfs[0], output_dir)
            else:
                print(f"No EDF files found in {path}")
    else:
        print(f"Cannot find data at {path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
