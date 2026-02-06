#!/usr/bin/env python3
"""
Φ-Dwell Grammar Decoder — Cracking the Eigenmode Language
==========================================================

Goes beyond the deep analyzer's attractor catalog to ask:
Does the brain's eigenmode grammar CHANGE between conditions?

Takes PhysioNet eegmmidb data where we have:
  R01 = Baseline (eyes open)
  R02 = Baseline (eyes closed)  
  R04, R08, R12 = Motor imagery (left/right hand, feet, fists)
  R03, R07, R11 = Motor execution

Extracts for each condition:
  1. VOCABULARY: Which 5-band eigenmode "words" does the brain use?
  2. SYNTAX: Transition probabilities between words (bigrams, trigrams)
  3. INFORMATION: Entropy, mutual information, predictability
  4. CONDITION SIGNATURES: Words that appear in task but not rest
  5. MOTOR ASYMMETRY: L-R eigenmode shifts during lateralized imagery

Usage:
    python phidwell_grammar_decoder.py "physionet.org/files/eegmmidb/1.0.0/" --subjects 5

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
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BANDS = {
    'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 50),
}
BAND_NAMES = list(BANDS.keys())
N_MODES = 8
MODE_NAMES = ['A-P', 'L-R', 'C-P', 'Diag', 'M5', 'M6', 'M7', 'M8']

# PhysioNet run labels
RUN_CONDITIONS = {
    'R01': 'rest_eo',      # Baseline eyes open
    'R02': 'rest_ec',      # Baseline eyes closed
    'R03': 'exec_lr',      # Motor execution left/right fist
    'R04': 'imag_lr',      # Motor imagery left/right fist
    'R05': 'exec_lr',      # Motor execution left/right fist (repeat)
    'R06': 'imag_lr',      # Motor imagery left/right fist (repeat)
    'R07': 'exec_bf',      # Motor execution both fists/feet
    'R08': 'imag_bf',      # Motor imagery both fists/feet
    'R09': 'exec_lr',      # Motor execution left/right fist (repeat)
    'R10': 'imag_lr',      # Motor imagery left/right fist (repeat)
    'R11': 'exec_bf',      # Motor execution both fists/feet (repeat)
    'R12': 'imag_bf',      # Motor imagery both fists/feet (repeat)
    'R13': 'exec_lr',
    'R14': 'imag_lr',
}

# Electrode positions (64-channel 10-10 system)
ELECTRODE_POS_64 = {
    'Fc5.': (-0.59, 0.31), 'Fc3.': (-0.35, 0.31), 'Fc1.': (-0.12, 0.31),
    'Fcz.': (0.0, 0.31), 'Fc2.': (0.12, 0.31), 'Fc4.': (0.35, 0.31),
    'Fc6.': (0.59, 0.31), 'C5..': (-0.67, 0.0), 'C3..': (-0.40, 0.0),
    'C1..': (-0.13, 0.0), 'Cz..': (0.0, 0.0), 'C2..': (0.13, 0.0),
    'C4..': (0.40, 0.0), 'C6..': (0.67, 0.0), 'Cp5.': (-0.59, -0.31),
    'Cp3.': (-0.35, -0.31), 'Cp1.': (-0.12, -0.31), 'Cpz.': (0.0, -0.31),
    'Cp2.': (0.12, -0.31), 'Cp4.': (0.35, -0.31), 'Cp6.': (0.59, -0.31),
    'Fp1.': (-0.25, 0.90), 'Fpz.': (0.0, 0.92), 'Fp2.': (0.25, 0.90),
    'Af7.': (-0.52, 0.75), 'Af3.': (-0.25, 0.72), 'Afz.': (0.0, 0.72),
    'Af4.': (0.25, 0.72), 'Af8.': (0.52, 0.75), 'F7..': (-0.73, 0.52),
    'F5..': (-0.55, 0.52), 'F3..': (-0.35, 0.52), 'F1..': (-0.12, 0.52),
    'Fz..': (0.0, 0.52), 'F2..': (0.12, 0.52), 'F4..': (0.35, 0.52),
    'F6..': (0.55, 0.52), 'F8..': (0.73, 0.52), 'Ft7.': (-0.80, 0.18),
    'Ft8.': (0.80, 0.18), 'T7..': (-0.85, 0.0), 'T8..': (0.85, 0.0),
    'T9..': (-0.95, 0.0), 'T10.': (0.95, 0.0), 'Tp7.': (-0.80, -0.18),
    'Tp8.': (0.80, -0.18), 'P7..': (-0.73, -0.52), 'P5..': (-0.55, -0.52),
    'P3..': (-0.35, -0.52), 'P1..': (-0.12, -0.52), 'Pz..': (0.0, -0.52),
    'P2..': (0.12, -0.52), 'P4..': (0.35, -0.52), 'P6..': (0.55, -0.52),
    'P8..': (0.73, -0.52), 'Po7.': (-0.45, -0.72), 'Po3.': (-0.25, -0.72),
    'Poz.': (0.0, -0.72), 'Po4.': (0.25, -0.72), 'Po8.': (0.45, -0.72),
    'O1..': (-0.25, -0.87), 'Oz..': (0.0, -0.87), 'O2..': (0.25, -0.87),
    'Iz..': (0.0, -0.95),
}


# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN EIGENMODES
# ═══════════════════════════════════════════════════════════════

def build_graph_laplacian(positions, sigma=0.5):
    """Build graph Laplacian and return eigenmodes."""
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    
    # Gaussian adjacency
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
    """Map EDF channel names to graph electrode names."""
    mapping = {}
    lookup = {}
    for gn in graph_names:
        clean = gn.replace('.', '').lower()
        lookup[clean] = gn
    
    for ch in raw_names:
        clean = ch.replace('EEG', '').strip().replace('.', '').replace(' ', '').lower()
        if clean in lookup:
            mapping[ch] = lookup[clean]
        else:
            # Fallback: strip trailing dots
            for eclean, gname in lookup.items():
                if clean.rstrip('.') == eclean or eclean.rstrip('.') == clean:
                    mapping[ch] = gname
                    break
    return mapping


# ═══════════════════════════════════════════════════════════════
# EEG → EIGENMODE WORD SEQUENCE
# ═══════════════════════════════════════════════════════════════

def eeg_to_words(filepath, graph_names, eigenvecs, word_step_ms=25):
    """
    Convert an EDF file into a sequence of eigenmode "words".
    
    Each word is a tuple of 5 values: (dom_mode_delta, dom_mode_theta, 
    dom_mode_alpha, dom_mode_beta, dom_mode_gamma) where each value
    is the index (0-7) of the dominant eigenmode in that band.
    
    Also returns the full coefficient timeseries and L-R asymmetry.
    """
    if not HAS_MNE:
        print("ERROR: mne required. pip install mne")
        return None
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose='error')
    mapping = map_channels(raw.ch_names, graph_names)
    
    if len(mapping) < 10:
        return None
    
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samp = raw.n_times
    n_elec = len(graph_names)
    
    # Band-filter and extract analytic signal phases
    phases = np.zeros((5, n_elec, n_samp), dtype=np.complex64)
    
    for bi, band in enumerate(BAND_NAMES):
        lo, hi = BANDS[band]
        sos = scipy.signal.butter(3, [lo, hi], btype='band', fs=sfreq, output='sos')
        
        for raw_ch, graph_ch in mapping.items():
            idx_g = graph_names.index(graph_ch)
            idx_r = raw.ch_names.index(raw_ch)
            sig = scipy.signal.sosfiltfilt(sos, data[idx_r])
            analytic = scipy.signal.hilbert(sig)
            phases[bi, idx_g, :] = analytic / (np.abs(analytic) + 1e-9)
    
    # Project onto eigenmodes: coeffs[band, mode, time]
    coeffs = np.abs(np.tensordot(phases, eigenvecs, axes=([1], [0])))
    # coeffs shape: (5, n_samp, N_MODES) → rearrange to (5, N_MODES, n_samp)
    # Actually tensordot gives (5, n_samp, N_MODES)
    
    # Dominant mode per band per timestep
    tokens = np.argmax(coeffs, axis=2)  # (5, n_samp) — mode index 0-7
    
    # Downsample to word_step_ms
    step = max(1, int(sfreq * word_step_ms / 1000))
    tokens_ds = tokens[:, ::step]  # (5, n_words)
    coeffs_ds = coeffs[:, ::step, :]  # (5, n_words, N_MODES)
    
    # Words: each column is a 5-tuple
    words = [tuple(tokens_ds[:, t]) for t in range(tokens_ds.shape[1])]
    
    # L-R asymmetry: Mode 1 (L-R) coefficient minus Mode 0 (A-P) coefficient
    # Specifically, the signed L-R eigenmode coefficient in beta band
    # (beta is motor-related, L-R asymmetry in beta = lateralized motor activity)
    lr_index = 1  # Mode 2 = L-R
    lr_asym_beta = coeffs_ds[3, :, lr_index]  # beta band, L-R mode magnitude
    lr_asym_alpha = coeffs_ds[2, :, lr_index]  # alpha band, L-R mode magnitude
    
    # Compute actual L-R signed asymmetry from the phase field
    # We need the signed eigenmode coefficient (not magnitude)
    signed_coeffs = np.tensordot(phases, eigenvecs, axes=([1], [0]))
    # signed_coeffs shape: (5, n_samp, N_MODES) — complex
    lr_signed_beta = np.real(signed_coeffs[3, ::step, lr_index])
    lr_signed_alpha = np.real(signed_coeffs[2, ::step, lr_index])
    
    return {
        'words': words,
        'coeffs': coeffs_ds,  # (5, n_words, N_MODES)
        'lr_signed_beta': lr_signed_beta,
        'lr_signed_alpha': lr_signed_alpha,
        'n_words': len(words),
        'sfreq': sfreq,
        'n_channels_mapped': len(mapping),
    }


# ═══════════════════════════════════════════════════════════════
# GRAMMAR ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_vocabulary(words):
    """Vocabulary statistics for a word sequence."""
    counts = Counter(words)
    total = len(words)
    vocab_size = len(counts)
    theoretical = N_MODES ** 5
    
    # Probabilities
    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    max_entropy = np.log2(vocab_size) if vocab_size > 1 else 1
    
    # Top words
    top = counts.most_common(30)
    
    # Concentration: fraction of time in top-5 words
    top5_frac = sum(c for _, c in counts.most_common(5)) / total
    
    # Zipf check: rank vs frequency on log-log
    ranks = np.arange(1, vocab_size + 1)
    freqs = np.array(sorted(counts.values(), reverse=True))
    
    # Fit log-log slope
    if vocab_size > 5:
        log_r = np.log(ranks[:min(50, vocab_size)])
        log_f = np.log(freqs[:min(50, vocab_size)])
        slope, _, r_value, _, _ = stats.linregress(log_r, log_f)
        zipf_exponent = -slope
        zipf_r2 = r_value ** 2
    else:
        zipf_exponent = 0
        zipf_r2 = 0
    
    return {
        'vocab_size': vocab_size,
        'theoretical': theoretical,
        'usage_pct': vocab_size / theoretical * 100,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'redundancy': 1 - entropy / max_entropy if max_entropy > 0 else 0,
        'top5_frac': top5_frac,
        'top_words': top,
        'zipf_exponent': zipf_exponent,
        'zipf_r2': zipf_r2,
        'ranks': ranks,
        'freqs': freqs,
        'counts': counts,
    }


def analyze_syntax(words):
    """Transition statistics: bigrams, trigrams, conditional entropy."""
    total = len(words)
    
    # Bigrams
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)
    
    # Non-self bigrams (actual transitions)
    transitions = [(a, b) for a, b in bigrams if a != b]
    trans_counts = Counter(transitions)
    
    # Self-transition rate
    n_self = sum(1 for a, b in bigrams if a == b)
    self_rate = n_self / len(bigrams) if bigrams else 0
    
    # Conditional entropy H(W_t+1 | W_t)
    # For each unique word, compute entropy of its successor distribution
    successor_dist = defaultdict(list)
    for a, b in bigrams:
        successor_dist[a].append(b)
    
    cond_entropy = 0
    word_counts = Counter(words[:-1])
    for w, successors in successor_dist.items():
        p_w = word_counts[w] / (total - 1)
        succ_counts = Counter(successors)
        succ_probs = np.array(list(succ_counts.values())) / len(successors)
        h_w = -np.sum(succ_probs * np.log2(succ_probs + 1e-15))
        cond_entropy += p_w * h_w
    
    # Mutual information: H(W_t) - H(W_t+1 | W_t)
    vocab = analyze_vocabulary(words)
    mi = vocab['entropy'] - cond_entropy
    
    # Trigrams (for detecting "phrases")
    trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
    trigram_counts = Counter(trigrams)
    
    # Non-trivial trigrams (not all same word)
    interesting_trigrams = {k: v for k, v in trigram_counts.items()
                           if not (k[0] == k[1] == k[2])}
    
    return {
        'self_rate': self_rate,
        'cond_entropy': cond_entropy,
        'mutual_info': mi,
        'predictability': mi / (vocab['entropy'] + 1e-15),
        'top_transitions': trans_counts.most_common(20),
        'top_trigrams': sorted(interesting_trigrams.items(), 
                               key=lambda x: x[1], reverse=True)[:15],
        'n_unique_transitions': len(trans_counts),
        'n_unique_trigrams': len(interesting_trigrams),
    }


def compare_vocabularies(vocab_rest, vocab_task, words_rest, words_task):
    """Find words unique to task, enriched in task, or depleted in task."""
    counts_rest = vocab_rest['counts']
    counts_task = vocab_task['counts']
    
    total_rest = sum(counts_rest.values())
    total_task = sum(counts_task.values())
    
    all_words = set(counts_rest.keys()) | set(counts_task.keys())
    
    # For each word, compute enrichment ratio
    enrichments = []
    for w in all_words:
        freq_rest = counts_rest.get(w, 0) / total_rest
        freq_task = counts_task.get(w, 0) / total_task
        
        # Log2 fold change (with pseudocount)
        pseudo = 1 / (total_rest + total_task)
        log2fc = np.log2((freq_task + pseudo) / (freq_rest + pseudo))
        
        # Chi-squared test for enrichment
        observed = np.array([counts_task.get(w, 0), counts_rest.get(w, 0)])
        expected_total = observed.sum()
        if expected_total > 5:  # Only test if enough observations
            expected = np.array([total_task, total_rest]) / (total_task + total_rest) * expected_total
            chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-9))
            p_val = 1 - stats.chi2.cdf(chi2, 1)
        else:
            chi2 = 0
            p_val = 1.0
        
        enrichments.append({
            'word': w,
            'freq_rest': freq_rest,
            'freq_task': freq_task,
            'log2fc': log2fc,
            'chi2': chi2,
            'p_val': p_val,
            'task_only': w not in counts_rest,
            'rest_only': w not in counts_task,
        })
    
    # Sort by absolute log2fc
    enrichments.sort(key=lambda x: abs(x['log2fc']), reverse=True)
    
    # Summary stats
    task_only = [e for e in enrichments if e['task_only']]
    rest_only = [e for e in enrichments if e['rest_only']]
    significant = [e for e in enrichments if e['p_val'] < 0.05]
    
    return {
        'enrichments': enrichments[:30],
        'n_task_only': len(task_only),
        'n_rest_only': len(rest_only),
        'n_significant': len(significant),
        'n_shared': len(all_words) - len(task_only) - len(rest_only),
        'jaccard': len(set(counts_rest.keys()) & set(counts_task.keys())) / 
                   len(all_words) if all_words else 0,
    }


def analyze_lr_asymmetry(result_rest, result_task):
    """Compare L-R eigenmode asymmetry between rest and task."""
    lr_rest = result_rest['lr_signed_beta']
    lr_task = result_task['lr_signed_beta']
    
    # Basic stats
    mean_rest = np.mean(lr_rest)
    mean_task = np.mean(lr_task)
    std_rest = np.std(lr_rest)
    std_task = np.std(lr_task)
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(lr_rest, lr_task)
    
    # Mann-Whitney
    try:
        mw_stat, mw_p = stats.mannwhitneyu(lr_rest, lr_task, alternative='two-sided')
    except:
        mw_stat, mw_p = 0, 1.0
    
    # Same for alpha
    lr_rest_a = result_rest['lr_signed_alpha']
    lr_task_a = result_task['lr_signed_alpha']
    ks_alpha, ks_p_alpha = stats.ks_2samp(lr_rest_a, lr_task_a)
    
    return {
        'beta_mean_rest': float(mean_rest),
        'beta_mean_task': float(mean_task),
        'beta_shift': float(mean_task - mean_rest),
        'beta_std_rest': float(std_rest),
        'beta_std_task': float(std_task),
        'beta_ks': float(ks_stat),
        'beta_ks_p': float(ks_p),
        'beta_mw_p': float(mw_p),
        'alpha_ks': float(ks_alpha),
        'alpha_ks_p': float(ks_p_alpha),
        'alpha_mean_rest': float(np.mean(lr_rest_a)),
        'alpha_mean_task': float(np.mean(lr_task_a)),
    }


# ═══════════════════════════════════════════════════════════════
# PER-BAND GRAMMAR (not just dominant mode — actual transition 
# matrices within each band separately)
# ═══════════════════════════════════════════════════════════════

def band_transition_matrices(words):
    """Extract 8x8 transition matrix per band from the word sequence."""
    if not words:
        return {}
    
    arr = np.array(words)  # (n_words, 5)
    results = {}
    
    for bi, band in enumerate(BAND_NAMES):
        modes = arr[:, bi]  # sequence of dominant modes for this band
        
        trans = np.zeros((N_MODES, N_MODES))
        for t in range(len(modes) - 1):
            trans[modes[t], modes[t + 1]] += 1
        
        row_sums = trans.sum(axis=1, keepdims=True)
        probs = np.divide(trans, row_sums, where=row_sums > 0,
                          out=np.zeros_like(trans))
        
        self_trans = np.diag(probs)
        
        results[band] = {
            'trans_matrix': trans,
            'trans_probs': probs,
            'self_trans': self_trans,
            'mean_self': float(np.mean(self_trans)),
        }
    
    return results


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_grammar_comparison(subject_id, rest_analysis, task_analysis, 
                            comparison, lr_asym, band_trans_rest, 
                            band_trans_task, output_path):
    """Giant comparison figure: REST vs TASK grammar."""
    
    fig = plt.figure(figsize=(28, 24), facecolor='#0a0a1a')
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.95, top=0.93, bottom=0.04)
    
    title_color = '#ff6090'
    text_color = '#d0d0e0'
    accent1 = '#50ffb4'  # rest
    accent2 = '#ff5050'  # task
    accent3 = '#ffb428'
    grid_color = '#1a1a3a'
    
    fig.suptitle(f'Φ-DWELL GRAMMAR DECODER  ·  {subject_id}  ·  REST vs TASK',
                 fontsize=20, color=title_color, fontweight='bold', y=0.97)
    
    def style_ax(ax, title=''):
        ax.set_facecolor('#0d0d24')
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')
        if title:
            ax.set_title(title, color=text_color, fontsize=11, fontweight='bold', pad=8)
    
    # ── Row 0: Vocabulary comparison ──
    
    # 0,0: Zipf plots (rank-frequency)
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'ZIPF DISTRIBUTION')
    vocab_r = rest_analysis['vocab']
    vocab_t = task_analysis['vocab']
    
    ax.loglog(vocab_r['ranks'], vocab_r['freqs'], 'o', color=accent1, 
              markersize=3, alpha=0.7, label=f"Rest (α={vocab_r['zipf_exponent']:.2f})")
    ax.loglog(vocab_t['ranks'], vocab_t['freqs'], 'o', color=accent2,
              markersize=3, alpha=0.7, label=f"Task (α={vocab_t['zipf_exponent']:.2f})")
    ax.set_xlabel('Rank', color=text_color, fontsize=9)
    ax.set_ylabel('Frequency', color=text_color, fontsize=9)
    ax.legend(fontsize=8, facecolor='#0d0d24', edgecolor='#2a2a4a', labelcolor=text_color)
    
    # 0,1: Vocabulary summary
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'VOCABULARY')
    ax.axis('off')
    
    info = [
        ['', 'REST', 'TASK'],
        ['Vocab size', f"{vocab_r['vocab_size']}", f"{vocab_t['vocab_size']}"],
        ['Usage %', f"{vocab_r['usage_pct']:.1f}%", f"{vocab_t['usage_pct']:.1f}%"],
        ['Entropy (bits)', f"{vocab_r['entropy']:.2f}", f"{vocab_t['entropy']:.2f}"],
        ['Redundancy', f"{vocab_r['redundancy']:.3f}", f"{vocab_t['redundancy']:.3f}"],
        ['Top-5 frac', f"{vocab_r['top5_frac']:.3f}", f"{vocab_t['top5_frac']:.3f}"],
        ['Zipf α', f"{vocab_r['zipf_exponent']:.2f}", f"{vocab_t['zipf_exponent']:.2f}"],
        ['Self-trans', f"{rest_analysis['syntax']['self_rate']:.3f}", 
                       f"{task_analysis['syntax']['self_rate']:.3f}"],
        ['Cond. H', f"{rest_analysis['syntax']['cond_entropy']:.2f}",
                    f"{task_analysis['syntax']['cond_entropy']:.2f}"],
        ['MI (bits)', f"{rest_analysis['syntax']['mutual_info']:.3f}",
                      f"{task_analysis['syntax']['mutual_info']:.3f}"],
        ['Predictability', f"{rest_analysis['syntax']['predictability']:.3f}",
                          f"{task_analysis['syntax']['predictability']:.3f}"],
    ]
    
    for i, row in enumerate(info):
        y = 0.95 - i * 0.085
        colors = [text_color, accent1, accent2]
        weights = ['bold' if i == 0 else 'normal'] * 3
        for j, (txt, col) in enumerate(zip(row, colors)):
            ax.text(0.05 + j * 0.35, y, txt, transform=ax.transAxes,
                    fontsize=9 if i > 0 else 10, color=col, fontweight=weights[j],
                    fontfamily='monospace')
    
    # 0,2: Jaccard & unique words
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, 'VOCABULARY OVERLAP')
    
    comp = comparison
    labels = ['Shared', 'Rest-only', 'Task-only', 'Significant\n(p<0.05)']
    values = [comp['n_shared'], comp['n_rest_only'], comp['n_task_only'], comp['n_significant']]
    colors = ['#8080c0', accent1, accent2, accent3]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.set_xlabel('Count', color=text_color, fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color=text_color, fontsize=9)
    ax.text(0.5, -0.15, f"Jaccard similarity: {comp['jaccard']:.3f}",
            transform=ax.transAxes, ha='center', color=accent3, fontsize=10)
    
    # 0,3: L-R asymmetry
    ax = fig.add_subplot(gs[0, 3])
    style_ax(ax, 'L-R EIGENMODE ASYMMETRY (β band)')
    
    bins = np.linspace(-0.5, 0.5, 50)
    ax.hist(lr_asym['lr_rest_beta'], bins=bins, alpha=0.5, color=accent1, 
            label='Rest', density=True)
    ax.hist(lr_asym['lr_task_beta'], bins=bins, alpha=0.5, color=accent2,
            label='Task', density=True)
    ax.axvline(lr_asym['stats']['beta_mean_rest'], color=accent1, ls='--', lw=2)
    ax.axvline(lr_asym['stats']['beta_mean_task'], color=accent2, ls='--', lw=2)
    ax.set_xlabel('Signed L-R coefficient', color=text_color, fontsize=9)
    ax.legend(fontsize=8, facecolor='#0d0d24', edgecolor='#2a2a4a', labelcolor=text_color)
    p_str = f"p={lr_asym['stats']['beta_ks_p']:.2e}" if lr_asym['stats']['beta_ks_p'] < 0.001 else f"p={lr_asym['stats']['beta_ks_p']:.4f}"
    ax.text(0.05, 0.95, f"KS: {p_str}", transform=ax.transAxes,
            color=accent3, fontsize=9, va='top')
    
    # ── Row 1: Per-band transition matrices comparison ──
    for bi, band in enumerate(BAND_NAMES):
        if bi < 4:
            ax = fig.add_subplot(gs[1, bi])
        else:
            ax = fig.add_subplot(gs[2, 0])
        
        # Difference matrix: task - rest
        diff = (band_trans_task[band]['trans_probs'] - 
                band_trans_rest[band]['trans_probs'])
        
        style_ax(ax, f'{band.upper()} Δ(task-rest)')
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.05, vmax=0.05, aspect='equal')
        ax.set_xticks(range(N_MODES))
        ax.set_yticks(range(N_MODES))
        ax.set_xticklabels(MODE_NAMES, fontsize=6, rotation=45)
        ax.set_yticklabels(MODE_NAMES, fontsize=6)
        
        # Annotate self-transitions
        for m in range(N_MODES):
            val = diff[m, m]
            color = '#ff0000' if val > 0.01 else '#0000ff' if val < -0.01 else '#808080'
            ax.text(m, m, f'{val:+.3f}', ha='center', va='center', 
                    fontsize=6, color=color, fontweight='bold')
    
    # ── Row 2: Top enriched/depleted words ──
    ax = fig.add_subplot(gs[2, 1:3])
    style_ax(ax, 'TOP ENRICHED/DEPLETED WORDS (Task vs Rest)')
    ax.axis('off')
    
    enriched = comparison['enrichments'][:20]
    headers = ['Word (δ θ α β γ)', 'Rest %', 'Task %', 'log₂FC', 'p-value']
    for j, h in enumerate(headers):
        ax.text(0.02 + j * 0.2, 0.97, h, transform=ax.transAxes,
                fontsize=8, color=accent3, fontweight='bold', fontfamily='monospace')
    
    for i, e in enumerate(enriched[:15]):
        y = 0.92 - i * 0.06
        word_str = ''.join(str(m + 1) for m in e['word'])
        fc_color = accent2 if e['log2fc'] > 0 else accent1
        sig = '*' if e['p_val'] < 0.05 else ''
        
        ax.text(0.02, y, f"  {word_str}", transform=ax.transAxes,
                fontsize=8, color=text_color, fontfamily='monospace')
        ax.text(0.22, y, f"{e['freq_rest']*100:.2f}%", transform=ax.transAxes,
                fontsize=8, color=accent1, fontfamily='monospace')
        ax.text(0.42, y, f"{e['freq_task']*100:.2f}%", transform=ax.transAxes,
                fontsize=8, color=accent2, fontfamily='monospace')
        ax.text(0.62, y, f"{e['log2fc']:+.2f}", transform=ax.transAxes,
                fontsize=8, color=fc_color, fontfamily='monospace', fontweight='bold')
        ax.text(0.82, y, f"{e['p_val']:.3f}{sig}", transform=ax.transAxes,
                fontsize=8, color=accent3 if sig else text_color, fontfamily='monospace')
    
    # ── Row 2,3: Band self-transition comparison ──
    ax = fig.add_subplot(gs[2, 3])
    style_ax(ax, 'SELF-TRANSITION RATE (per band)')
    
    x = np.arange(5)
    width = 0.35
    rest_self = [band_trans_rest[b]['mean_self'] for b in BAND_NAMES]
    task_self = [band_trans_task[b]['mean_self'] for b in BAND_NAMES]
    
    ax.bar(x - width/2, rest_self, width, color=accent1, alpha=0.8, label='Rest')
    ax.bar(x + width/2, task_self, width, color=accent2, alpha=0.8, label='Task')
    ax.set_xticks(x)
    ax.set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'], color=text_color, fontsize=11)
    ax.set_ylabel('Self-transition prob', color=text_color, fontsize=9)
    ax.legend(fontsize=8, facecolor='#0d0d24', edgecolor='#2a2a4a', labelcolor=text_color)
    
    # ── Row 3: Top words side by side ──
    ax = fig.add_subplot(gs[3, 0:2])
    style_ax(ax, 'TOP 10 WORDS: REST vs TASK')
    ax.axis('off')
    
    # Rest top words
    ax.text(0.02, 0.97, 'REST', transform=ax.transAxes, fontsize=11,
            color=accent1, fontweight='bold')
    ax.text(0.52, 0.97, 'TASK', transform=ax.transAxes, fontsize=11,
            color=accent2, fontweight='bold')
    
    rest_top = vocab_r['top_words'][:10]
    task_top = vocab_t['top_words'][:10]
    total_r = sum(c for _, c in vocab_r['top_words'])
    total_t = sum(c for _, c in vocab_t['top_words'])
    
    for i, ((wr, cr), (wt, ct)) in enumerate(zip(rest_top, task_top)):
        y = 0.90 - i * 0.085
        wr_str = ''.join(str(m+1) for m in wr)
        wt_str = ''.join(str(m+1) for m in wt)
        
        # Decode: show mode names
        def decode_word(w):
            parts = []
            for bi, mi in enumerate(w):
                if mi < 2:  # Only show the interesting modes
                    parts.append(f"{BAND_NAMES[bi][0].upper()}:{MODE_NAMES[mi]}")
            return ' '.join(parts) if parts else 'mixed'
        
        ax.text(0.02, y, f"{i+1}. {wr_str}", transform=ax.transAxes,
                fontsize=9, color=accent1, fontfamily='monospace')
        ax.text(0.15, y, f"{cr/len(rest_analysis['words'])*100:.1f}%", 
                transform=ax.transAxes, fontsize=8, color=text_color, fontfamily='monospace')
        ax.text(0.25, y, decode_word(wr), transform=ax.transAxes,
                fontsize=7, color='#8080b0', fontfamily='monospace')
        
        ax.text(0.52, y, f"{i+1}. {wt_str}", transform=ax.transAxes,
                fontsize=9, color=accent2, fontfamily='monospace')
        ax.text(0.65, y, f"{ct/len(task_analysis['words'])*100:.1f}%",
                transform=ax.transAxes, fontsize=8, color=text_color, fontfamily='monospace')
        ax.text(0.75, y, decode_word(wt), transform=ax.transAxes,
                fontsize=7, color='#8080b0', fontfamily='monospace')
    
    # ── Row 3: Trigram phrases ──
    ax = fig.add_subplot(gs[3, 2:4])
    style_ax(ax, 'TOP "PHRASES" (Non-trivial trigrams)')
    ax.axis('off')
    
    ax.text(0.02, 0.97, 'REST', transform=ax.transAxes, fontsize=10,
            color=accent1, fontweight='bold')
    ax.text(0.52, 0.97, 'TASK', transform=ax.transAxes, fontsize=10,
            color=accent2, fontweight='bold')
    
    rest_tri = rest_analysis['syntax']['top_trigrams'][:8]
    task_tri = task_analysis['syntax']['top_trigrams'][:8]
    
    for i in range(max(len(rest_tri), len(task_tri))):
        y = 0.89 - i * 0.11
        
        if i < len(rest_tri):
            tri, cnt = rest_tri[i]
            s = ' → '.join(''.join(str(m+1) for m in w) for w in tri)
            ax.text(0.02, y, f"{s}  ×{cnt}", transform=ax.transAxes,
                    fontsize=7, color=accent1, fontfamily='monospace')
        
        if i < len(task_tri):
            tri, cnt = task_tri[i]
            s = ' → '.join(''.join(str(m+1) for m in w) for w in tri)
            ax.text(0.52, y, f"{s}  ×{cnt}", transform=ax.transAxes,
                    fontsize=7, color=accent2, fontfamily='monospace')
    
    # ── Row 4: Summary / verdict ──
    ax = fig.add_subplot(gs[4, :])
    style_ax(ax, '')
    ax.axis('off')
    
    # Compute key differences
    entropy_diff = vocab_t['entropy'] - vocab_r['entropy']
    vocab_diff = vocab_t['vocab_size'] - vocab_r['vocab_size']
    self_diff = task_analysis['syntax']['self_rate'] - rest_analysis['syntax']['self_rate']
    mi_diff = task_analysis['syntax']['mutual_info'] - rest_analysis['syntax']['mutual_info']
    
    summary_lines = [
        f"ENTROPY: Task {'higher' if entropy_diff > 0 else 'lower'} by {abs(entropy_diff):.3f} bits "
        f"({'more diverse' if entropy_diff > 0 else 'more concentrated'} vocabulary during task)",
        f"VOCABULARY: Task uses {abs(vocab_diff)} {'more' if vocab_diff > 0 else 'fewer'} unique words "
        f"({comp['n_task_only']} task-only, {comp['n_rest_only']} rest-only, Jaccard={comp['jaccard']:.3f})",
        f"PERSISTENCE: Task self-transition {'higher' if self_diff > 0 else 'lower'} by {abs(self_diff):.4f} "
        f"(brain {'dwells longer' if self_diff > 0 else 'switches faster'} during task)",
        f"PREDICTABILITY: Task MI {'higher' if mi_diff > 0 else 'lower'} by {abs(mi_diff):.4f} bits "
        f"(transitions {'more structured' if mi_diff > 0 else 'more random'} during task)",
        f"L-R ASYMMETRY: Beta L-R shift = {lr_asym['stats']['beta_shift']:+.4f} (KS p={lr_asym['stats']['beta_ks_p']:.4f}) "
        f"{'** SIGNIFICANT **' if lr_asym['stats']['beta_ks_p'] < 0.05 else ''}",
        f"SIGNIFICANT WORDS: {comp['n_significant']} words change frequency significantly (p<0.05) between conditions",
    ]
    
    for i, line in enumerate(summary_lines):
        ax.text(0.02, 0.85 - i * 0.15, line, transform=ax.transAxes,
                fontsize=10, color=text_color if i < 5 else accent3, 
                fontfamily='monospace')
    
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def process_subject(subject_dir, graph_names, eigenvecs, output_dir):
    """Process one subject: extract rest and task grammars, compare."""
    subject_id = os.path.basename(subject_dir.rstrip('/\\'))
    print(f"\n{'═' * 60}")
    print(f"  Processing {subject_id}")
    print(f"{'═' * 60}")
    
    # Find all EDF files
    edfs = sorted(glob.glob(os.path.join(subject_dir, '*.edf')))
    if not edfs:
        print(f"  No EDF files found in {subject_dir}")
        return None
    
    # Categorize runs
    rest_files = []
    task_files = []
    
    for f in edfs:
        basename = os.path.basename(f)
        # Extract run number: S001R01.edf → R01
        parts = basename.replace('.edf', '').split('R')
        if len(parts) < 2:
            continue
        run_id = 'R' + parts[-1].zfill(2)
        
        cond = RUN_CONDITIONS.get(run_id, None)
        if cond is None:
            continue
        
        if cond.startswith('rest'):
            rest_files.append(f)
        elif cond.startswith('imag'):  # Focus on imagery (most interesting)
            task_files.append(f)
    
    if not rest_files or not task_files:
        print(f"  Need both rest and task files. Found {len(rest_files)} rest, {len(task_files)} task")
        return None
    
    print(f"  Rest files: {len(rest_files)}, Task files: {len(task_files)}")
    
    # Process all files
    rest_words = []
    task_words = []
    rest_results = []
    task_results = []
    
    for f in rest_files[:2]:  # Max 2 rest runs
        result = eeg_to_words(f, graph_names, eigenvecs)
        if result:
            rest_words.extend(result['words'])
            rest_results.append(result)
            print(f"    REST: {os.path.basename(f)} → {result['n_words']} words, "
                  f"{result['n_channels_mapped']}ch")
    
    for f in task_files[:4]:  # Max 4 task runs
        result = eeg_to_words(f, graph_names, eigenvecs)
        if result:
            task_words.extend(result['words'])
            task_results.append(result)
            print(f"    TASK: {os.path.basename(f)} → {result['n_words']} words, "
                  f"{result['n_channels_mapped']}ch")
    
    if not rest_words or not task_words:
        print(f"  Failed to extract words")
        return None
    
    print(f"\n  Total: {len(rest_words)} rest words, {len(task_words)} task words")
    
    # Analyze
    print(f"  Analyzing vocabulary...")
    vocab_rest = analyze_vocabulary(rest_words)
    vocab_task = analyze_vocabulary(task_words)
    
    print(f"  Analyzing syntax...")
    syntax_rest = analyze_syntax(rest_words)
    syntax_task = analyze_syntax(task_words)
    
    print(f"  Comparing vocabularies...")
    comparison = compare_vocabularies(vocab_rest, vocab_task, rest_words, task_words)
    
    print(f"  Computing L-R asymmetry...")
    # Merge L-R data
    lr_rest_beta = np.concatenate([r['lr_signed_beta'] for r in rest_results])
    lr_task_beta = np.concatenate([r['lr_signed_beta'] for r in task_results])
    lr_rest_alpha = np.concatenate([r['lr_signed_alpha'] for r in rest_results])
    lr_task_alpha = np.concatenate([r['lr_signed_alpha'] for r in task_results])
    
    merged_rest = {'lr_signed_beta': lr_rest_beta, 'lr_signed_alpha': lr_rest_alpha}
    merged_task = {'lr_signed_beta': lr_task_beta, 'lr_signed_alpha': lr_task_alpha}
    lr_stats = analyze_lr_asymmetry(merged_rest, merged_task)
    
    print(f"  Computing per-band transitions...")
    band_trans_rest = band_transition_matrices(rest_words)
    band_trans_task = band_transition_matrices(task_words)
    
    # Package results
    rest_analysis = {
        'vocab': vocab_rest,
        'syntax': syntax_rest,
        'words': rest_words,
    }
    task_analysis = {
        'vocab': vocab_task,
        'syntax': syntax_task,
        'words': task_words,
    }
    
    lr_data = {
        'lr_rest_beta': lr_rest_beta,
        'lr_task_beta': lr_task_beta,
        'stats': lr_stats,
    }
    
    # Print key findings
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  GRAMMAR COMPARISON: {subject_id:>10}          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  {'':20} {'REST':>8}  {'TASK':>8}  │")
    print(f"  │  {'Vocabulary size':20} {vocab_rest['vocab_size']:>8}  {vocab_task['vocab_size']:>8}  │")
    print(f"  │  {'Entropy (bits)':20} {vocab_rest['entropy']:>8.3f}  {vocab_task['entropy']:>8.3f}  │")
    print(f"  │  {'Self-transition':20} {syntax_rest['self_rate']:>8.4f}  {syntax_task['self_rate']:>8.4f}  │")
    print(f"  │  {'Mutual info':20} {syntax_rest['mutual_info']:>8.4f}  {syntax_task['mutual_info']:>8.4f}  │")
    print(f"  │  {'Predictability':20} {syntax_rest['predictability']:>8.4f}  {syntax_task['predictability']:>8.4f}  │")
    print(f"  │  {'Jaccard overlap':20} {comparison['jaccard']:>8.3f}           │")
    print(f"  │  {'Task-only words':20} {comparison['n_task_only']:>8}           │")
    print(f"  │  {'Significant (p<.05)':20} {comparison['n_significant']:>8}           │")
    print(f"  │  {'L-R β shift':20} {lr_stats['beta_shift']:>+8.4f}           │")
    print(f"  │  {'L-R β KS p':20} {lr_stats['beta_ks_p']:>8.4f}           │")
    print(f"  └─────────────────────────────────────────┘")
    
    # Per-band self-transition comparison
    print(f"\n  Per-band self-transition (rest → task):")
    for band in BAND_NAMES:
        sr = band_trans_rest[band]['mean_self']
        st = band_trans_task[band]['mean_self']
        diff = st - sr
        arrow = '↑' if diff > 0.002 else '↓' if diff < -0.002 else '='
        print(f"    {band:>6}: {sr:.4f} → {st:.4f}  ({diff:+.4f}) {arrow}")
    
    # Visualize
    fig_path = os.path.join(output_dir, f'phidwell_grammar_{subject_id}.png')
    plot_grammar_comparison(subject_id, rest_analysis, task_analysis,
                           comparison, lr_data, band_trans_rest, 
                           band_trans_task, fig_path)
    
    # Save JSON
    json_path = os.path.join(output_dir, f'phidwell_grammar_{subject_id}.json')
    json_data = {
        'subject': subject_id,
        'rest_vocab_size': vocab_rest['vocab_size'],
        'task_vocab_size': vocab_task['vocab_size'],
        'rest_entropy': vocab_rest['entropy'],
        'task_entropy': vocab_task['entropy'],
        'rest_self_rate': syntax_rest['self_rate'],
        'task_self_rate': syntax_task['self_rate'],
        'rest_mi': syntax_rest['mutual_info'],
        'task_mi': syntax_task['mutual_info'],
        'rest_predictability': syntax_rest['predictability'],
        'task_predictability': syntax_task['predictability'],
        'jaccard': comparison['jaccard'],
        'n_task_only': comparison['n_task_only'],
        'n_rest_only': comparison['n_rest_only'],
        'n_significant': comparison['n_significant'],
        'lr_beta_shift': lr_stats['beta_shift'],
        'lr_beta_ks_p': lr_stats['beta_ks_p'],
        'lr_alpha_ks_p': lr_stats['alpha_ks_p'],
        'rest_zipf': vocab_rest['zipf_exponent'],
        'task_zipf': vocab_task['zipf_exponent'],
        'band_self_rest': {b: band_trans_rest[b]['mean_self'] for b in BAND_NAMES},
        'band_self_task': {b: band_trans_task[b]['mean_self'] for b in BAND_NAMES},
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")
    
    return json_data


def main():
    parser = argparse.ArgumentParser(description='Φ-Dwell Grammar Decoder')
    parser.add_argument('path', help='Path to eegmmidb directory or single subject')
    parser.add_argument('--subjects', type=int, default=5, help='Number of subjects')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════╗")
    print("║   Φ-DWELL GRAMMAR DECODER                      ║")
    print("║   Cracking the Eigenmode Language               ║")
    print("║   REST vs MOTOR IMAGERY                         ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Build graph Laplacian
    print("\nBuilding graph Laplacian eigenmodes...")
    graph_names, coords, eigenvecs, eigenvals = build_graph_laplacian(ELECTRODE_POS_64)
    print(f"  {len(graph_names)} electrodes, {N_MODES} eigenmodes")
    print(f"  Mode 1 (λ={eigenvals[0]:.2f}): A-P dipole")
    print(f"  Mode 2 (λ={eigenvals[1]:.2f}): L-R hemispheric")
    
    # Find subjects
    path = args.path
    output_dir = path if os.path.isdir(path) else os.path.dirname(path)
    
    if os.path.isdir(path):
        # Check if this is a subject directory or the root
        edfs = glob.glob(os.path.join(path, '*.edf'))
        if edfs:
            # Single subject
            subjects = [path]
        else:
            # Root directory — find subject subdirectories
            subjects = sorted(glob.glob(os.path.join(path, 'S*')))
            subjects = [s for s in subjects if os.path.isdir(s)]
            subjects = subjects[:args.subjects]
    else:
        subjects = [os.path.dirname(path)]
    
    print(f"\nFound {len(subjects)} subjects to process")
    
    all_results = []
    for subj_dir in subjects:
        result = process_subject(subj_dir, graph_names, eigenvecs, output_dir)
        if result:
            all_results.append(result)
    
    # Cross-subject summary
    if len(all_results) > 1:
        print(f"\n{'═' * 60}")
        print(f"  CROSS-SUBJECT SUMMARY ({len(all_results)} subjects)")
        print(f"{'═' * 60}")
        
        # Aggregate
        for key in ['rest_entropy', 'task_entropy', 'rest_self_rate', 'task_self_rate',
                     'rest_mi', 'task_mi', 'jaccard', 'n_significant', 'lr_beta_ks_p']:
            vals = [r[key] for r in all_results]
            print(f"  {key:>25}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
        # Consistent directions
        entropy_up = sum(1 for r in all_results if r['task_entropy'] > r['rest_entropy'])
        self_up = sum(1 for r in all_results if r['task_self_rate'] > r['rest_self_rate'])
        mi_up = sum(1 for r in all_results if r['task_mi'] > r['rest_mi'])
        lr_sig = sum(1 for r in all_results if r['lr_beta_ks_p'] < 0.05)
        
        n = len(all_results)
        print(f"\n  Direction consistency ({n} subjects):")
        print(f"    Entropy higher in task:     {entropy_up}/{n}")
        print(f"    Self-trans higher in task:   {self_up}/{n}")
        print(f"    MI higher in task:           {mi_up}/{n}")
        print(f"    L-R β significant (p<.05):   {lr_sig}/{n}")
        
        # Per-band direction
        print(f"\n  Per-band self-transition shift (task - rest):")
        for band in BAND_NAMES:
            diffs = [r['band_self_task'][band] - r['band_self_rest'][band] 
                     for r in all_results]
            mean_d = np.mean(diffs)
            # Sign test
            n_pos = sum(1 for d in diffs if d > 0)
            from scipy.stats import binomtest
            try:
                bt = binomtest(n_pos, n, 0.5)
                p = bt.pvalue
            except:
                p = 1.0
            arrow = '↑' if mean_d > 0.001 else '↓' if mean_d < -0.001 else '='
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {band:>6}: {mean_d:+.4f} {arrow}  ({n_pos}/{n} positive) p={p:.4f} {sig}")
        
        # Save all results
        all_path = os.path.join(output_dir, 'phidwell_grammar_all.json')
        with open(all_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved: {all_path}")


if __name__ == '__main__':
    main()