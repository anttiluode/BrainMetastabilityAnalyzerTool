"""
BRAIN GRAMMAR EXTRACTOR
=======================
Treats Brain Dynamics as a Language.

1. TOKENIZATION: Converts continuous EEG into discrete "Neural Words".
   - At every time t, the state is a tuple: (Mode_Delta, Mode_Theta, Mode_Alpha, Mode_Beta, Mode_Gamma).
   - Theoretical Vocabulary: 8^5 = 32,768 possible states.
   - Actual Vocabulary: The subset of states the brain actually uses.

2. SYNTAX ANALYSIS:
   - Bigrams: Which state follows which? (Transition Matrix)
   - N-grams: Common "phrases" (sequences of 3+ states).
   - Entropy: How predictable is the brain's language?

Usage:
    python brain_grammar.py "physionet.org/files/eegmmidb/1.0.0/S001/"
"""

import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt
import argparse
import sys
import os
import glob
import collections
import networkx as nx
from tabulate import tabulate

try:
    import mne
except ImportError:
    print("ERROR: mne required. pip install mne")
    sys.exit(1)

# --- CONFIG ---
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
N_MODES = 8  # Use first 8 eigenmodes

# Electrodes (64-ch 10-10 system for PhysioNet)
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

def build_graph(positions, sigma=0.5):
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((coords[i] - coords[j])**2))
            w = np.exp(-d**2 / (2*sigma**2))
            A[i,j] = A[j,i] = w
    D = np.diag(A.sum(axis=1))
    L = D - A
    vals, vecs = scipy.linalg.eigh(L)
    return names, vecs[:, 1:N_MODES+1]  # Skip mode 0

def map_channels(raw_names, graph_names):
    mapping = {}
    clean_graph = {n.replace('.', '').lower(): n for n in graph_names}
    for ch in raw_names:
        clean = ch.replace('EEG', '').strip().replace('.', '').lower()
        if clean in clean_graph:
            mapping[ch] = clean_graph[clean]
    return mapping

class BrainGrammar:
    def __init__(self):
        self.names, self.vecs = build_graph(ELECTRODE_POS_64)
        
    def process_file(self, path):
        raw = mne.io.read_raw_edf(path, preload=True, verbose='error')
        mapping = map_channels(raw.ch_names, self.names)
        if len(mapping) < 10: return None
        
        # 1. Extract Phases for all bands
        # Shape: [n_bands, n_electrodes, n_samples]
        n_samp = raw.n_times
        sfreq = raw.info['sfreq']
        phases = np.zeros((5, len(self.names), n_samp), dtype=np.complex64)
        
        data = raw.get_data()
        
        for bi, band in enumerate(BAND_ORDER):
            lo, hi = BANDS[band]
            # Fast filter
            sos = scipy.signal.butter(3, [lo, hi], btype='band', fs=sfreq, output='sos')
            
            for raw_ch, graph_ch in mapping.items():
                idx_g = self.names.index(graph_ch)
                idx_r = raw.ch_names.index(raw_ch)
                
                sig = scipy.signal.sosfiltfilt(sos, data[idx_r])
                # Hilbert
                analytic = scipy.signal.hilbert(sig)
                # Normalize to unit circle immediately to save memory
                phases[bi, idx_g, :] = analytic / (np.abs(analytic) + 1e-9)

        # 2. Project to Eigenmodes
        # Coeffs: [n_bands, n_modes, n_samples]
        # We only care about magnitude for dominance
        coeffs = np.abs(np.tensordot(phases, self.vecs, axes=([1], [0])))
        
        # 3. Tokenize: Find dominant mode for each band at each time step
        # Tokens: [n_bands, n_samples] -> values 0..7 (representing modes 1..8)
        tokens = np.argmax(coeffs, axis=1) # (5, n_samples)
        
        # 4. Collapse to Words
        # A "Word" is a tuple of 5 modes: (m_delta, m_theta, m_alpha, m_beta, m_gamma)
        # We downsample to ~25ms steps (40Hz) to avoid micro-jitter
        step = int(sfreq * 0.025) 
        words = tokens[:, ::step].T # (n_steps, 5)
        
        return words

def analyze_grammar(all_words):
    # Flatten list of arrays
    if not all_words: return
    corpus = np.concatenate(all_words, axis=0)
    
    # 1. Vocabulary Analysis
    # Convert rows to unique string IDs for counting
    word_ids = [tuple(row) for row in corpus]
    counts = collections.Counter(word_ids)
    
    vocab_size = len(counts)
    theoretical = N_MODES ** 5
    usage = vocab_size / theoretical * 100
    
    # Top 10 Words
    top_words = counts.most_common(10)
    
    # 2. Syntax (Bigrams)
    # P(Word_B | Word_A)
    # We look for the most common transitions
    bigrams = list(zip(word_ids[:-1], word_ids[1:]))
    bigram_counts = collections.Counter(bigrams)
    top_transitions = bigram_counts.most_common(10)
    
    # 3. Entropy
    probs = np.array(list(counts.values())) / len(corpus)
    entropy = -np.sum(probs * np.log2(probs))
    
    print("\n" + "="*60)
    print("THE LANGUAGE OF THE BRAIN")
    print("="*60)
    print(f"Total Time Steps:   {len(corpus)}")
    print(f"Unique States Used: {vocab_size} / {theoretical} ({usage:.2f}%)")
    print(f"Shannon Entropy:    {entropy:.2f} bits (Max: {np.log2(vocab_size):.2f})")
    
    print("\nTOP 10 'WORDS' (Dominant Modes: Δ θ α β γ)")
    print("-" * 45)
    headers = ["Rank", "Delta", "Theta", "Alpha", "Beta", "Gamma", "Freq %", "Description"]
    table = []
    
    # Heuristic description
    def describe(w):
        # w is (d, t, a, b, g) indices 0-7
        # Mode 0 (Index 0, Eigenmode 1) = A-P Dipole
        # Mode 1 (Index 1, Eigenmode 2) = L-R Dipole
        if w[0] == 0 and w[1] == 0: return "Deep A-P Axis"
        if w[2] == 1: return "Alpha L-R Split"
        if w[0] == w[1] == w[2]: return "Broadband Sync"
        return "Mixed State"

    for i, (word, count) in enumerate(top_words):
        # Convert 0-7 to 1-8 for display
        display_word = [x+1 for x in word] 
        perc = count / len(corpus) * 100
        table.append([i+1, *display_word, f"{perc:.1f}%", describe(word)])
        
    print(tabulate(table, headers=headers))
    
    print("\nTOP TRANSITIONS (Syntax)")
    print("-" * 45)
    t_table = []
    for i, ((w1, w2), count) in enumerate(top_transitions):
        if w1 == w2: continue # Skip self-loops
        d1 = "".join(str(x+1) for x in w1)
        d2 = "".join(str(x+1) for x in w2)
        perc = count / len(bigrams) * 100
        t_table.append([f"{d1} -> {d2}", f"{perc:.2f}%"])
        if len(t_table) >= 10: break
        
    print(tabulate(t_table, headers=["Transition", "Probability"]))

    # 4. Generate State Graph
    G = nx.DiGraph()
    # Add top 20 nodes
    top_nodes = [w for w, c in counts.most_common(20)]
    for w in top_nodes:
        G.add_node(w, count=counts[w])
    
    # Add edges between top nodes
    for (w1, w2), count in bigram_counts.items():
        if w1 in top_nodes and w2 in top_nodes and w1 != w2:
            if count > len(corpus) * 0.001: # Filter weak links
                G.add_edge(w1, w2, weight=count)
    
    print(f"\nGenerated State Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to subject folder (e.g. S001)')
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.path, "*.edf"))
    files = sorted(files)[:4] # Analyze first 4 runs (Base + Tasks)
    
    bg = BrainGrammar()
    all_corpus = []
    
    print(f"Reading {len(files)} files...")
    for f in files:
        words = bg.process_file(f)
        if words is not None:
            all_corpus.append(words)
            
    if all_corpus:
        G = analyze_grammar(all_corpus)