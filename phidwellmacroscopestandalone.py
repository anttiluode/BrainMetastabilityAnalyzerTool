#!/usr/bin/env python3
"""
Φ-Dwell Macroscope — Standalone Real-Time Brain Dynamics Viewer
================================================================

A real-time eigenmode phase portrait of brain dynamics from scalp EEG.

Loads an EDF file (or PhysioNet eegmmidb directory), decomposes the
holographic EEG phase field into graph Laplacian eigenmodes at each
time step, and renders a live dashboard showing:

  ┌─────────────────────┬───────────────┐
  │                     │  REGIME       │
  │   PHASE PORTRAIT    │  EIGENMODES   │
  │   (2D orbit +       │  DWELL TIMER  │
  │    attractor map)   │  BAND POWER   │
  │                     │  METRICS      │
  ├─────────┬───────────┼───────────────┤
  │ DWELL   │ MODE 1    │ BAND POWER   │
  │ DISTRIB │ TRAJECTORY│ HISTORY      │
  └─────────┴───────────┴───────────────┘

Theory:
  Biström & Claude (2025). Φ-Dwell: Eigenmode Phase-Field Metastability.
  Wang et al. (2017). Brain network eigenmodes.
  Baker & Cariani (2025). Time-domain account of brain function.
  Vollan et al. (2025). Theta sweeps in entorhinal-hippocampal maps.

Usage:
  python phidwell_macroscope_standalone.py path/to/file.edf
  python phidwell_macroscope_standalone.py path/to/eegmmidb/1.0.0/S001/ --band theta
  python phidwell_macroscope_standalone.py path/to/eegmmidb/1.0.0/S001/ --band alpha --speed 4
  python phidwell_macroscope_standalone.py --demo   (synthetic demo mode, no EEG needed)

Requirements:
  pip install numpy scipy matplotlib mne
"""

import numpy as np
import scipy.signal
import scipy.linalg
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from collections import deque
import argparse
import sys
import os
import glob
import time
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_COLORS = {
    'delta':  '#ff5050',
    'theta':  '#ffb428',
    'alpha':  '#50ff50',
    'beta':   '#50a0ff',
    'gamma':  '#c850ff',
}

MODE_NAMES = ['A-P', 'L-R', 'C-P', 'Diag', 'M5', 'M6', 'M7', 'M8']

REGIME_COLORS = {
    'critical':  '#50ffb4',
    'bursty':    '#ffb428',
    'clocklike': '#50a0ff',
    'random':    '#787878',
}

# 10-20 electrode positions (normalized)
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

# 10-10 electrode positions (64 channel, PhysioNet eegmmidb)
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


# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN
# ═══════════════════════════════════════════════════════════════

def build_graph_laplacian(positions, sigma=0.5):
    """Build graph Laplacian from electrode positions."""
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])

    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 +
                        (coords[i, 1] - coords[j, 1])**2)
            A[i, j] = np.exp(-d**2 / (2 * sigma**2))
            A[j, i] = A[i, j]

    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = scipy.linalg.eigh(L)

    return names, coords, eigvals, eigvecs, L


def map_channels(raw_ch_names, electrode_names):
    """Map EDF channel names to electrode position names."""
    mapping = {}
    clean_lookup = {}
    for ename in electrode_names:
        clean = ename.replace('.', '').lower()
        clean_lookup[clean] = ename

    for ch in raw_ch_names:
        clean = (ch.replace('EEG ', '').replace('EEG', '')
                 .replace('-REF', '').replace('-Ref', '').replace('-ref', '')
                 .strip().replace(' ', '').lower()
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
# EEG DATA LOADER
# ═══════════════════════════════════════════════════════════════

class EEGDataSource:
    """Loads and pre-processes EEG data for real-time playback."""

    def __init__(self, path, speed=1.0):
        import mne
        self.speed = speed

        # Find EDF file(s)
        if os.path.isfile(path) and path.lower().endswith('.edf'):
            edf_path = path
        elif os.path.isdir(path):
            edfs = sorted(glob.glob(os.path.join(path, '*R01*.edf')))
            if not edfs:
                edfs = sorted(glob.glob(os.path.join(path, '*.edf')))
            if not edfs:
                raise FileNotFoundError(f"No EDF files found in {path}")
            edf_path = edfs[0]
        else:
            raise FileNotFoundError(f"Cannot find EEG data at {path}")

        print(f"Loading: {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
        self.sfreq = raw.info['sfreq']
        self.data = raw.get_data()
        self.ch_names = raw.ch_names
        self.n_samples = self.data.shape[1]
        self.duration = self.n_samples / self.sfreq

        # Detect electrode system
        n_eeg = sum(1 for ch in self.ch_names
                    if not ch.startswith('STI') and 'annotation' not in ch.lower())
        if n_eeg >= 60:
            self.positions = ELECTRODE_POS_64
            self.system = '10-10 (64ch)'
        else:
            self.positions = ELECTRODE_POS_20
            self.system = '10-20 (20ch)'

        # Build graph
        self.e_names, self.e_coords, self.eigvals, self.eigvecs, self.L = \
            build_graph_laplacian(self.positions)

        # Map channels
        self.mapping = map_channels(self.ch_names, self.e_names)
        print(f"  System: {self.system}, mapped {len(self.mapping)}/{len(self.e_names)} channels")
        print(f"  Duration: {self.duration:.1f}s, sfreq: {self.sfreq}Hz")

        # Pre-compute phases for all bands
        self.band_phases = {}
        for band_name, (lo, hi) in BANDS.items():
            print(f"  Filtering {band_name} ({lo}-{hi}Hz)...", end=' ', flush=True)
            nyq = self.sfreq / 2.0
            lo_n = max(0.5, lo) / nyq
            hi_n = min(hi, nyq - 1) / nyq
            b, a = scipy.signal.butter(3, [lo_n, hi_n], btype='band')

            phases = np.zeros((len(self.e_names), self.n_samples), dtype=np.float32)
            active = np.zeros(len(self.e_names), dtype=bool)

            for ch_raw, ch_graph in self.mapping.items():
                idx_g = self.e_names.index(ch_graph)
                idx_r = self.ch_names.index(ch_raw)
                filtered = scipy.signal.filtfilt(b, a, self.data[idx_r])
                analytic = scipy.signal.hilbert(filtered)
                phases[idx_g] = np.angle(analytic).astype(np.float32)
                active[idx_g] = True

            self.band_phases[band_name] = phases
            self.active_mask = active
            print("done")

        # Playback state
        self.current_sample = 0
        self.step_size = max(1, int(self.sfreq / 30))  # ~30fps base

        # Eigenvectors (skip mode 0 = constant)
        self.n_modes = 8
        self.V = self.eigvecs[:, 1:self.n_modes + 1].copy()
        self.V[~self.active_mask] = 0
        for m in range(self.n_modes):
            norm = np.linalg.norm(self.V[:, m])
            if norm > 1e-10:
                self.V[:, m] /= norm

        print(f"  Eigenmodes ready: {self.n_modes} modes")
        print(f"  Eigenvalues: {', '.join(f'{v:.2f}' for v in self.eigvals[1:self.n_modes+1])}")

    def get_eigenmode_coefficients(self, sample_idx):
        """
        Extract eigenmode coefficients for all 5 bands at a given sample.
        
        Returns:
            all_coeffs: (5, n_modes) coefficient magnitudes
            all_phases: (5, n_modes) coefficient phases
            band_powers: (5,) total power per band
        """
        # Window: take circular mean of phases in a small window
        win = int(self.sfreq * 0.05)  # 50ms window
        start = max(0, sample_idx - win // 2)
        end = min(self.n_samples, start + win)

        all_coeffs = np.zeros((5, self.n_modes))
        all_phases = np.zeros((5, self.n_modes))
        band_powers = np.zeros(5)

        for bi, band in enumerate(BAND_NAMES):
            phases = self.band_phases[band]

            # Circular mean phase per electrode
            sin_mean = np.mean(np.sin(phases[:, start:end]), axis=1)
            cos_mean = np.mean(np.cos(phases[:, start:end]), axis=1)
            mean_phase = np.arctan2(sin_mean, cos_mean)

            z_real = np.cos(mean_phase)
            z_imag = np.sin(mean_phase)

            for m in range(self.n_modes):
                pr = np.dot(z_real, self.V[:, m])
                pi = np.dot(z_imag, self.V[:, m])
                all_coeffs[bi, m] = np.sqrt(pr**2 + pi**2)
                all_phases[bi, m] = np.arctan2(pi, pr)

            band_powers[bi] = np.sum(all_coeffs[bi])

        return all_coeffs, all_phases, band_powers

    def advance(self):
        """Advance playback by one frame. Returns current time in seconds."""
        step = int(self.step_size * self.speed)
        self.current_sample = (self.current_sample + step) % self.n_samples
        return self.current_sample / self.sfreq


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DEMO SOURCE
# ═══════════════════════════════════════════════════════════════

class DemoDataSource:
    """Synthetic data for demo mode — simulates brain dynamics."""

    def __init__(self):
        self.sfreq = 160
        self.n_modes = 8
        self.positions = ELECTRODE_POS_20
        self.system = 'DEMO (synthetic)'
        self.duration = float('inf')
        self.current_sample = 0
        self.step_size = 5

        self.e_names, self.e_coords, self.eigvals, self.eigvecs, self.L = \
            build_graph_laplacian(self.positions)
        self.V = self.eigvecs[:, 1:self.n_modes + 1].copy()
        for m in range(self.n_modes):
            norm = np.linalg.norm(self.V[:, m])
            if norm > 1e-10:
                self.V[:, m] /= norm

        # Oscillators for each band×mode
        self.oscillators = []
        for b in range(5):
            for m in range(self.n_modes):
                self.oscillators.append({
                    'phase': np.random.rand() * 2 * np.pi,
                    'freq': 0.5 + b * 0.8 + m * 0.3 + np.random.rand() * 0.5,
                    'amp': 0.3 + 0.5 * np.exp(-m / 3) * np.exp(-abs(b - 2) / 2),
                    'drift': 0,
                })

        self.state = 'eyes_open'
        self.t = 0

        # State profiles
        self.profiles = {
            'eyes_open':  {'band_bias': [0.15, 0.35, 0.15, 0.20, 0.15], 'noise': 0.04},
            'eyes_closed': {'band_bias': [0.15, 0.10, 0.45, 0.15, 0.15], 'noise': 0.025},
            'meditation':  {'band_bias': [0.35, 0.30, 0.20, 0.10, 0.05], 'noise': 0.012},
        }

        print("DEMO MODE — synthetic brain dynamics")
        print("  Press 1=EyesOpen  2=EyesClosed  3=Meditation in the plot window")

    def get_eigenmode_coefficients(self, sample_idx=None):
        profile = self.profiles.get(self.state, self.profiles['eyes_open'])
        dt = 1 / 30

        all_coeffs = np.zeros((5, self.n_modes))
        all_phases = np.zeros((5, self.n_modes))
        band_powers = np.zeros(5)

        for b in range(5):
            for m in range(self.n_modes):
                idx = b * self.n_modes + m
                osc = self.oscillators[idx]

                target_amp = (0.3 + 0.6 * np.exp(-m / 3)) * profile['band_bias'][b] * 3
                osc['amp'] += (target_amp - osc['amp']) * 0.05
                osc['phase'] += osc['freq'] * dt * (1 + b * 0.5)
                osc['drift'] += (np.random.randn() * 0.005)
                osc['drift'] *= 0.99
                osc['phase'] += osc['drift']

                if np.random.rand() < profile['noise'] * dt * (1 + b * 0.3):
                    osc['phase'] += np.random.randn() * np.pi * (0.3 + b * 0.15)

                all_coeffs[b, m] = osc['amp'] * (0.5 + 0.5 * np.cos(osc['phase']))
                all_phases[b, m] = osc['phase'] % (2 * np.pi)

            band_powers[b] = np.sum(all_coeffs[b])

        self.t += dt
        return all_coeffs, all_phases, band_powers

    def advance(self):
        self.current_sample += self.step_size
        return self.t


# ═══════════════════════════════════════════════════════════════
# MACROSCOPE ENGINE
# ═══════════════════════════════════════════════════════════════

class MacroscopeEngine:
    """Core analysis engine — tracks trajectories, detects regimes, computes metrics."""

    def __init__(self, n_modes=8):
        self.n_modes = n_modes

        # Trajectory
        self.max_trail = 600
        self.trail_x = deque(maxlen=self.max_trail)
        self.trail_y = deque(maxlen=self.max_trail)
        self.trail_band = deque(maxlen=self.max_trail)

        # Coefficient history
        self.coeff_history = deque(maxlen=300)
        self.phase_history = deque(maxlen=300)

        # Dwell tracking
        self.current_dwell = 0
        self.last_phase_vec = None
        self.dwell_times = deque(maxlen=500)
        self.dwell_threshold = np.pi / 4
        self.dwell_histogram = np.zeros(40)

        # Attractor heatmap
        self.heat_size = 100
        self.heatmap = np.zeros((self.heat_size, self.heat_size))
        self.heat_decay = 0.997

        # PCA projection
        self.pca_axes = None
        self.pca_center = None
        self.pca_scale = 1.0

        # Metrics
        self.regime = 'random'
        self.regime_conf = 0.0
        self.running_cv = 0.0
        self.metastability = 0.0
        self.dominant_band = 0
        self.dominant_mode = 0
        self.current_dwell_ms = 0

        # History for charts
        self.mode1_history = deque(maxlen=200)
        self.band_history = deque(maxlen=200)

        self.frame = 0

    def step(self, all_coeffs, all_phases, band_powers, fps=30):
        """Process one frame of eigenmode data."""
        self.frame += 1

        # Dominant band and mode
        self.dominant_band = int(np.argmax(band_powers))
        dom_coeffs = all_coeffs[self.dominant_band]
        self.dominant_mode = int(np.argmax(dom_coeffs))

        # Full state vector
        full_coeffs = all_coeffs.flatten()
        self.coeff_history.append(full_coeffs.copy())
        self.phase_history.append(all_phases.flatten().copy())

        # Dwell detection
        phase_vec = all_phases[self.dominant_band]
        if self.last_phase_vec is not None:
            diffs = np.abs(phase_vec - self.last_phase_vec)
            diffs = np.minimum(diffs, 2 * np.pi - diffs)
            max_jump = np.max(diffs)

            if max_jump < self.dwell_threshold:
                self.current_dwell += 1
            else:
                if self.current_dwell > 0:
                    self.dwell_times.append(self.current_dwell)
                    bin_idx = min(39, self.current_dwell // 2)
                    self.dwell_histogram[bin_idx] += 1
                self.current_dwell = 1
        else:
            self.current_dwell = 1
        self.last_phase_vec = phase_vec.copy()
        self.current_dwell_ms = int(self.current_dwell * (1000.0 / max(fps, 1)))

        # Regime detection (every 10 frames)
        if self.frame % 10 == 0:
            self._detect_regime()
            self._compute_metastability()

        # Update PCA (every 30 frames)
        if self.frame % 30 == 0 and len(self.coeff_history) >= 30:
            self._update_pca()

        # Project to 2D
        px, py = self._project_2d(full_coeffs)
        self.trail_x.append(px)
        self.trail_y.append(py)
        self.trail_band.append(self.dominant_band)

        # Heatmap
        hx = int(np.clip((px + 2) / 4 * self.heat_size, 0, self.heat_size - 1))
        hy = int(np.clip((2 - py) / 4 * self.heat_size, 0, self.heat_size - 1))
        self.heatmap *= self.heat_decay
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < self.heat_size and 0 <= ny < self.heat_size:
                    self.heatmap[ny, nx] += 0.3 * np.exp(-(dx**2 + dy**2) / 3)

        # Chart histories
        self.mode1_history.append(all_coeffs[self.dominant_band, 0])
        self.band_history.append(band_powers.copy())

    def _detect_regime(self):
        if len(self.dwell_times) < 10:
            self.regime = 'random'
            self.regime_conf = 0
            return
        arr = np.array(list(self.dwell_times)[-100:])
        mean = np.mean(arr)
        if mean < 1e-6:
            self.regime = 'random'
            return
        std = np.std(arr)
        cv = std / mean
        self.running_cv = cv
        kurt = stats.kurtosis(arr) if len(arr) > 4 else 0

        if cv < 0.7:
            self.regime = 'clocklike'
            self.regime_conf = min(1, (0.7 - cv) / 0.3)
        elif cv > 1.3 and kurt > 80:
            self.regime = 'bursty'
            self.regime_conf = min(1, (kurt - 80) / 200)
        elif cv > 1.1 and 15 < kurt < 100:
            self.regime = 'critical'
            self.regime_conf = min(1, (cv - 1.1) / 0.5)
        else:
            self.regime = 'random'
            self.regime_conf = 0.5

    def _compute_metastability(self):
        if len(self.dwell_times) < 5:
            self.metastability = 0
            return
        arr = np.array(list(self.dwell_times)[-100:])
        mean = np.mean(arr)
        if mean < 1e-6:
            self.metastability = 0
            return
        cv = np.std(arr) / mean
        cv_score = max(0, 1 - abs(cv - 1.3) / 1.3)
        max_ratio = np.max(arr) / mean
        tail_score = min(1, max_ratio / 10)
        self.metastability = 0.5 * cv_score + 0.5 * tail_score

    def _update_pca(self):
        data = np.array(self.coeff_history)
        center = np.mean(data, axis=0)
        centered = data - center
        cov = np.dot(centered.T, centered) / len(centered)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            self.pca_axes = eigvecs[:, idx[:2]].T
            self.pca_center = center
            projected = np.dot(centered, self.pca_axes.T)
            self.pca_scale = max(np.std(projected) * 3, 0.01)
        except Exception:
            pass

    def _project_2d(self, full_coeffs):
        if self.pca_axes is None or self.pca_center is None:
            return float(full_coeffs[0] - 0.5), float(full_coeffs[1] - 0.5)
        centered = full_coeffs - self.pca_center
        p = np.dot(self.pca_axes, centered) / self.pca_scale
        return float(p[0]), float(p[1])


# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB DASHBOARD
# ═══════════════════════════════════════════════════════════════

class MacroscopeDashboard:
    """Real-time matplotlib dashboard for the macroscope."""

    def __init__(self, source, engine, band='alpha', speed=1.0):
        self.source = source
        self.engine = engine
        self.band = band
        self.speed = speed
        self.paused = False
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()

        # Figure setup
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='#0a0a12')
        self.fig.canvas.manager.set_window_title('Φ-Dwell Macroscope')

        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.35,
                      left=0.05, right=0.97, top=0.92, bottom=0.06)

        # Main phase portrait (large)
        self.ax_portrait = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_portrait.set_facecolor('#0a0a12')

        # Eigenmode bar chart
        self.ax_modes = self.fig.add_subplot(gs[0, 2])
        self.ax_modes.set_facecolor('#0f0f1a')

        # Band power bars
        self.ax_bands = self.fig.add_subplot(gs[0, 3])
        self.ax_bands.set_facecolor('#0f0f1a')

        # Regime / metrics text panel
        self.ax_info = self.fig.add_subplot(gs[1, 2:4])
        self.ax_info.set_facecolor('#0f0f1a')

        # Dwell distribution
        self.ax_dwell = self.fig.add_subplot(gs[2, 0])
        self.ax_dwell.set_facecolor('#0f0f1a')

        # Mode 1 trajectory
        self.ax_traj = self.fig.add_subplot(gs[2, 1])
        self.ax_traj.set_facecolor('#0f0f1a')

        # Band power history
        self.ax_bphist = self.fig.add_subplot(gs[2, 2:4])
        self.ax_bphist.set_facecolor('#0f0f1a')

        # Title
        self.fig.suptitle(
            f'Φ-DWELL MACROSCOPE   ·   {source.system}   ·   [{band.upper()}]',
            fontsize=13, fontweight='bold', color='#b44adc',
            fontfamily='monospace', y=0.97
        )

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == '1' and isinstance(self.source, DemoDataSource):
            self.source.state = 'eyes_open'
        elif event.key == '2' and isinstance(self.source, DemoDataSource):
            self.source.state = 'eyes_closed'
        elif event.key == '3' and isinstance(self.source, DemoDataSource):
            self.source.state = 'meditation'

    def update(self, frame_num):
        """Animation update function."""
        if self.paused:
            return

        now = time.time()
        self.fps_counter.append(now - self.last_time)
        self.last_time = now
        fps = 1.0 / max(np.mean(self.fps_counter), 0.001)

        # Advance source and get data
        t_sec = self.source.advance()
        all_coeffs, all_phases, band_powers = self.source.get_eigenmode_coefficients(
            getattr(self.source, 'current_sample', 0))

        # Step engine
        self.engine.step(all_coeffs, all_phases, band_powers, fps)

        # ── Render all panels ──
        self._render_portrait()
        self._render_modes(all_coeffs)
        self._render_bands(band_powers)
        self._render_info(t_sec, fps)
        self._render_dwell_hist()
        self._render_trajectory()
        self._render_band_history()

    def _render_portrait(self):
        ax = self.ax_portrait
        ax.clear()
        ax.set_facecolor('#0a0a12')

        # Heatmap background
        if np.max(self.engine.heatmap) > 0:
            extent = [-2, 2, -2, 2]
            ax.imshow(self.engine.heatmap, extent=extent, origin='upper',
                      cmap='inferno', alpha=0.5, aspect='auto',
                      vmin=0, vmax=max(np.max(self.engine.heatmap) * 0.8, 0.01))

        # Crosshairs
        ax.axhline(0, color='#2a2a3a', linewidth=0.5)
        ax.axvline(0, color='#2a2a3a', linewidth=0.5)

        # Trail
        n = len(self.engine.trail_x)
        if n > 1:
            xs = list(self.engine.trail_x)
            ys = list(self.engine.trail_y)
            bands = list(self.engine.trail_band)

            # Draw segments colored by band, with fading
            for i in range(max(0, n - 400), n - 1):
                alpha = max(0.05, 1.0 - (n - i) / 400)
                color = BAND_COLORS[BAND_NAMES[bands[i+1]]]
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                        color=color, alpha=alpha, linewidth=1.2 if i > n-10 else 0.7)

            # Current position
            bc = BAND_COLORS[BAND_NAMES[self.engine.dominant_band]]
            ax.plot(xs[-1], ys[-1], 'o', color=bc, markersize=8, zorder=10)
            ax.plot(xs[-1], ys[-1], 'o', color='white', markersize=10,
                    fillstyle='none', linewidth=1.5, zorder=10)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('PC1', fontsize=8, color='#5a5a7a')
        ax.set_ylabel('PC2', fontsize=8, color='#5a5a7a')
        ax.set_title('EIGENMODE PHASE PORTRAIT', fontsize=9, color='#8888aa',
                      fontfamily='monospace')
        ax.tick_params(colors='#3a3a5a', labelsize=7)

    def _render_modes(self, all_coeffs):
        ax = self.ax_modes
        ax.clear()
        ax.set_facecolor('#0f0f1a')

        dom_coeffs = all_coeffs[self.engine.dominant_band]
        max_c = max(np.max(dom_coeffs), 0.01)
        x = np.arange(self.engine.n_modes)
        colors = ['#ffffff' if i == self.engine.dominant_mode else '#3a4a6a'
                  for i in range(self.engine.n_modes)]
        ax.bar(x, dom_coeffs / max_c, color=colors, width=0.7, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(MODE_NAMES[:self.engine.n_modes], fontsize=6, color='#5a5a7a')
        ax.set_ylim(0, 1.2)
        ax.set_title('EIGENMODES', fontsize=8, color='#8888aa', fontfamily='monospace')
        ax.tick_params(left=False, labelleft=False, colors='#3a3a5a')

    def _render_bands(self, band_powers):
        ax = self.ax_bands
        ax.clear()
        ax.set_facecolor('#0f0f1a')

        max_p = max(np.max(band_powers), 0.01)
        x = np.arange(5)
        colors = [BAND_COLORS[b] for b in BAND_NAMES]
        alphas = [1.0 if i == self.engine.dominant_band else 0.4 for i in range(5)]
        for i in range(5):
            ax.bar(i, band_powers[i] / max_p, color=colors[i], alpha=alphas[i], width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['δ', 'θ', 'α', 'β', 'γ'], fontsize=9, color='#8888aa')
        ax.set_ylim(0, 1.2)
        ax.set_title('BAND POWER', fontsize=8, color='#8888aa', fontfamily='monospace')
        ax.tick_params(left=False, labelleft=False, colors='#3a3a5a')

    def _render_info(self, t_sec, fps):
        ax = self.ax_info
        ax.clear()
        ax.set_facecolor('#0f0f1a')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        rc = REGIME_COLORS.get(self.engine.regime, '#787878')

        # Regime
        ax.text(0.5, 9.0, 'REGIME', fontsize=8, color='#5a5a7a',
                fontfamily='monospace', fontweight='bold')
        ax.text(0.5, 7.2, self.engine.regime.upper(), fontsize=22, color=rc,
                fontfamily='monospace', fontweight='bold')

        # Confidence bar
        ax.barh(6.3, self.engine.regime_conf * 4, height=0.25, color=rc, alpha=0.7)
        ax.barh(6.3, 4, height=0.25, color='#2a2a3a', alpha=0.3)

        # Metrics column
        mx = 5.5
        metrics = [
            ('DWELL',         f'{self.engine.current_dwell_ms}ms'),
            ('CV',            f'{self.engine.running_cv:.2f}'),
            ('METASTABILITY', f'{self.engine.metastability:.2f}'),
            ('DOM. MODE',     MODE_NAMES[min(self.engine.dominant_mode, 7)]),
            ('DOM. BAND',     BAND_NAMES[self.engine.dominant_band].upper()),
            ('TIME',          f'{t_sec:.1f}s'),
            ('FPS',           f'{fps:.0f}'),
        ]

        for i, (key, val) in enumerate(metrics):
            y = 9.2 - i * 1.25
            ax.text(mx, y, key, fontsize=7, color='#5a5a7a', fontfamily='monospace')
            val_color = '#c8c8e8'
            if key == 'DWELL' and self.engine.current_dwell_ms > 100:
                val_color = '#50ff50'
            elif key == 'CV' and self.engine.running_cv > 1.0:
                val_color = '#ffb428'
            ax.text(mx + 3.2, y, val, fontsize=9, color=val_color,
                    fontfamily='monospace', fontweight='bold')

    def _render_dwell_hist(self):
        ax = self.ax_dwell
        ax.clear()
        ax.set_facecolor('#0f0f1a')

        hist = self.engine.dwell_histogram
        if np.sum(hist) > 0:
            x = np.arange(len(hist))
            rc = REGIME_COLORS.get(self.engine.regime, '#787878')
            ax.bar(x, hist, color=rc, alpha=0.6, width=0.9)

        ax.set_title('DWELL DISTRIBUTION', fontsize=8, color='#8888aa', fontfamily='monospace')
        ax.set_xlabel('Duration (frames)', fontsize=7, color='#5a5a7a')
        ax.tick_params(colors='#3a3a5a', labelsize=6)

    def _render_trajectory(self):
        ax = self.ax_traj
        ax.clear()
        ax.set_facecolor('#0f0f1a')

        data = list(self.engine.mode1_history)
        if len(data) > 1:
            bc = BAND_COLORS[BAND_NAMES[self.engine.dominant_band]]
            ax.plot(data, color=bc, linewidth=0.8, alpha=0.8)

        ax.set_title('MODE 1 (A-P) COEFFICIENT', fontsize=8, color='#8888aa',
                      fontfamily='monospace')
        ax.tick_params(colors='#3a3a5a', labelsize=6)

    def _render_band_history(self):
        ax = self.ax_bphist
        ax.clear()
        ax.set_facecolor('#0f0f1a')

        data = list(self.engine.band_history)
        if len(data) > 1:
            arr = np.array(data)
            for b in range(5):
                alpha = 1.0 if b == self.engine.dominant_band else 0.25
                ax.plot(arr[:, b], color=BAND_COLORS[BAND_NAMES[b]],
                        linewidth=1.0 if alpha > 0.5 else 0.6, alpha=alpha)

        ax.set_title('BAND POWER HISTORY', fontsize=8, color='#8888aa', fontfamily='monospace')
        ax.tick_params(colors='#3a3a5a', labelsize=6)
        ax.legend(['δ', 'θ', 'α', 'β', 'γ'], fontsize=6, ncol=5,
                  loc='upper right', framealpha=0.3)

    def run(self, interval_ms=33):
        """Start the animation loop."""
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval_ms, blit=False, cache_frame_data=False)
        print("\n┌──────────────────────────────────────────────┐")
        print("│  Φ-DWELL MACROSCOPE — RUNNING                │")
        print("│  SPACE = pause/resume                        │")
        if isinstance(self.source, DemoDataSource):
            print("│  1 = Eyes Open  2 = Eyes Closed  3 = Meditate│")
        print("│  Close window to exit                        │")
        print("└──────────────────────────────────────────────┘")
        plt.show()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Φ-Dwell Macroscope — Real-time eigenmode phase portrait of brain dynamics"
    )
    parser.add_argument('data_path', nargs='?', default=None,
                        help='Path to EDF file or eegmmidb subject directory')
    parser.add_argument('--band', default='alpha',
                        choices=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        help='Primary frequency band to analyze (default: alpha)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with synthetic data (no EEG needed)')
    parser.add_argument('--interval', type=int, default=33,
                        help='Animation interval in ms (default: 33 = ~30fps)')

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   Φ-DWELL MACROSCOPE                             ║")
    print("║   Eigenmode Phase Portrait of Brain Dynamics     ║")
    print("║   Biström & Claude (2025)                        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    if args.demo or args.data_path is None:
        source = DemoDataSource()
    else:
        source = EEGDataSource(args.data_path, speed=args.speed)

    engine = MacroscopeEngine(n_modes=8)
    dashboard = MacroscopeDashboard(source, engine, band=args.band, speed=args.speed)
    dashboard.run(interval_ms=args.interval)


if __name__ == '__main__':
    main()