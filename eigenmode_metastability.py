"""
Eigenmode Metastability Analyzer
=================================
Bridges Pribram's holographic brain → Wang et al. spectral graph theory → Baker & Cariani temporal coding.

Instead of sweeping an arbitrary k parameter, we:
1. Build the graph Laplacian of the electrode array (64ch 10-10 or 20ch 10-20)
2. Compute its eigenmodes (the actual spatial basis functions)
3. At each time window, project the EEG phase field onto each eigenmode
4. Track eigenmode coefficient trajectories over time
5. Compute dwell-time distributions per eigenmode
6. Compare eyes-open vs eyes-closed, live vs pink noise

The key insight: k in the holographic field ≈ eigenmode index.
k=1 → eigenmode 2 (hemispheric dipole). k→high → higher spatial modes.
But eigenmodes are the CORRECT basis, not an arbitrary continuous parameter.

PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb):
- 109 subjects, 64 channels, 160 Hz
- Run 1: eyes open baseline (1 min)
- Run 2: eyes closed baseline (1 min)

Usage:
    python eigenmode_metastability.py /path/to/eegmmidb/S001/
    python eigenmode_metastability.py /path/to/eegmmidb/ --subjects 10 --band alpha
    python eigenmode_metastability.py /path/to/eegmmidb/ --all
"""

import numpy as np
import scipy.signal
import scipy.linalg
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import sys
import os
import time
import glob
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import mne
except ImportError:
    print("ERROR: mne required. pip install mne")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 10-10 ELECTRODE POSITIONS (64 channels, BCI2000 / PhysioNet eegmmidb)
# Normalized to unit circle. Positions from standard 10-10 layout.
# ═══════════════════════════════════════════════════════════════

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

# Fallback: standard 10-20 positions for 20-channel EEGs
ELECTRODE_POS_20 = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6),
    'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
    'T7': (-0.9, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0, 0.0),
    'C4': (0.4, 0.0), 'T8': (0.9, 0.0),
    'P7': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5),
    'P4': (0.35, -0.5), 'P8': (0.7, -0.5),
    'O1': (-0.3, -0.85), 'Oz': (0, -0.85), 'O2': (0.3, -0.85)
}

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}


# ═══════════════════════════════════════════════════════════════
# GRAPH LAPLACIAN & EIGENMODES
# ═══════════════════════════════════════════════════════════════

def build_electrode_graph(positions, sigma=0.5):
    """
    Build adjacency matrix from electrode positions using Gaussian kernel.
    A_ij = exp(-d_ij^2 / (2*sigma^2))

    This creates a graph where nearby electrodes are strongly connected,
    distant ones weakly. The Laplacian eigenmodes of this graph are the
    spatial basis functions — analogous to the structural connectome
    eigenmodes of Wang et al., but at the scalp electrode level.

    Args:
        positions: dict of {name: (x, y)}
        sigma: Gaussian kernel width (controls connectivity falloff)

    Returns:
        names: list of electrode names
        coords: (N, 2) array of positions
        eigvals: (N,) eigenvalues of graph Laplacian
        eigvecs: (N, N) eigenvectors (columns = eigenmodes)
        L: (N, N) graph Laplacian
    """
    names = sorted(positions.keys())
    N = len(names)
    coords = np.array([positions[n] for n in names])

    # Adjacency matrix via Gaussian kernel
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 +
                        (coords[i, 1] - coords[j, 1])**2)
            A[i, j] = np.exp(-d**2 / (2 * sigma**2))
            A[j, i] = A[i, j]

    # Degree matrix
    D = np.diag(A.sum(axis=1))

    # Graph Laplacian: L = D - A
    L = D - A

    # Eigen-decomposition (symmetric → real eigenvalues)
    eigvals, eigvecs = scipy.linalg.eigh(L)

    return names, coords, eigvals, eigvecs, L


def map_channels_to_electrodes(raw_ch_names, electrode_names):
    """
    Map EDF channel names to electrode position names.
    Handles BCI2000/PhysioNet naming conventions.
    """
    mapping = {}  # raw_name → electrode_name

    # Build lookup from cleaned name → electrode name
    clean_lookup = {}
    for ename in electrode_names:
        # Strip trailing dots used for padding
        clean = ename.replace('.', '').lower()
        clean_lookup[clean] = ename

    for ch in raw_ch_names:
        # Clean the raw channel name
        clean = (ch.replace('EEG ', '').replace('EEG', '')
                 .replace('-REF', '').replace('-Ref', '').replace('-ref', '')
                 .strip().replace(' ', '').lower()
                 .replace('t3', 't7').replace('t4', 't8')
                 .replace('t5', 'p7').replace('t6', 'p8'))

        if clean in clean_lookup:
            mapping[ch] = clean_lookup[clean]
        else:
            # Try variations
            for ename in electrode_names:
                eclean = ename.replace('.', '').lower()
                if clean == eclean or clean.rstrip('.') == eclean:
                    mapping[ch] = ename
                    break

    return mapping


# ═══════════════════════════════════════════════════════════════
# CORE ANALYZER
# ═══════════════════════════════════════════════════════════════

class EigenmodeMetastabilityAnalyzer:
    """
    Projects EEG phase fields onto graph Laplacian eigenmodes
    and analyzes temporal dwell statistics per eigenmode.
    """

    def __init__(self, electrode_positions=None, band='alpha',
                 window_ms=250, step_ms=10, n_eigenmodes=10,
                 stability_threshold=0.15):
        """
        Args:
            electrode_positions: dict of {name: (x, y)} or None for auto-detect
            band: frequency band
            window_ms: sliding window width
            step_ms: step between windows
            n_eigenmodes: number of eigenmodes to track (skip mode 0 = constant)
            stability_threshold: fractional change in eigenmode coefficient
                                 to count as "stable" (relative to std)
        """
        self.band = band
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.n_modes = n_eigenmodes
        self.stability_threshold = stability_threshold
        self.electrode_positions = electrode_positions

        # Will be set during load
        self.e_names = None
        self.e_coords = None
        self.eigvals = None
        self.eigvecs = None
        self.L = None

    def _init_graph(self, positions):
        """Build graph from electrode positions."""
        self.e_names, self.e_coords, self.eigvals, self.eigvecs, self.L = \
            build_electrode_graph(positions)
        print(f"  Graph: {len(self.e_names)} electrodes, "
              f"σ={0.5}, {self.n_modes} eigenmodes")
        print(f"  Eigenvalues (modes 1-{self.n_modes}): "
              f"{', '.join(f'{v:.2f}' for v in self.eigvals[1:self.n_modes+1])}")

    def load_edf(self, path):
        """Load EDF and extract band-filtered phases mapped to graph nodes."""
        print(f"Loading: {os.path.basename(path)}")
        raw = mne.io.read_raw_edf(path, preload=True, verbose='error')
        sfreq = raw.info['sfreq']

        # Decide which electrode set to use
        n_eeg = sum(1 for ch in raw.ch_names
                     if not ch.startswith('STI') and not ch.startswith('Status')
                     and 'annotation' not in ch.lower())

        if self.electrode_positions is not None:
            positions = self.electrode_positions
        elif n_eeg >= 60:
            positions = ELECTRODE_POS_64
        else:
            positions = ELECTRODE_POS_20

        if self.e_names is None:
            self._init_graph(positions)

        # Map channels
        mapping = map_channels_to_electrodes(raw.ch_names, self.e_names)
        print(f"  Mapped {len(mapping)}/{len(self.e_names)} channels, "
              f"sfreq={sfreq}Hz, dur={raw.times[-1]:.1f}s")

        if len(mapping) < 10:
            print(f"  WARNING: Only {len(mapping)} channels mapped!")

        # Bandpass filter and extract phase
        lo, hi = BANDS[self.band]
        nyq = sfreq / 2.0
        lo_n = max(0.5, lo) / nyq
        hi_n = min(hi, nyq - 1) / nyq
        b, a = scipy.signal.butter(3, [lo_n, hi_n], btype='band')

        data = raw.get_data()
        n_samples = data.shape[1]

        # Phase array: (n_electrodes, n_samples)
        # Order matches self.e_names
        phases = np.zeros((len(self.e_names), n_samples), dtype=np.float32)
        active_mask = np.zeros(len(self.e_names), dtype=bool)

        for ch_raw, ch_graph in mapping.items():
            idx_graph = self.e_names.index(ch_graph)
            idx_raw = raw.ch_names.index(ch_raw)
            filtered = scipy.signal.filtfilt(b, a, data[idx_raw])
            analytic = scipy.signal.hilbert(filtered)
            phases[idx_graph] = np.angle(analytic).astype(np.float32)
            active_mask[idx_graph] = True

        return phases, active_mask, sfreq, n_samples

    def generate_pink_noise_phases(self, n_electrodes, n_samples, sfreq):
        """Generate band-filtered pink noise control (no cross-channel correlation)."""
        lo, hi = BANDS[self.band]
        nyq = sfreq / 2.0
        lo_n = max(0.5, lo) / nyq
        hi_n = min(hi, nyq - 1) / nyq
        b, a = scipy.signal.butter(3, [lo_n, hi_n], btype='band')

        phases = np.zeros((n_electrodes, n_samples), dtype=np.float32)
        for i in range(n_electrodes):
            white = np.random.randn(n_samples)
            X = np.fft.rfft(white)
            S = np.arange(len(X)) + 1
            pink = np.fft.irfft(X / S, n=n_samples)
            filtered = scipy.signal.filtfilt(b, a, pink)
            analytic = scipy.signal.hilbert(filtered)
            phases[i] = np.angle(analytic).astype(np.float32)

        return phases

    def compute_eigenmode_timeseries(self, phases, active_mask, sfreq, n_samples,
                                      label=""):
        """
        Project phase field onto graph eigenmodes at each time window.

        At each window:
        1. Compute circular mean phase per electrode
        2. Form complex field: z_i = exp(j * phase_i)
        3. Project real and imaginary parts onto each eigenvector
        4. Eigenmode coefficient = magnitude of projection

        Returns:
            times: (n_windows,) center times
            coefficients: (n_modes, n_windows) eigenmode coefficient magnitudes
            phases_proj: (n_modes, n_windows) eigenmode phase angles
        """
        window_samp = int(self.window_ms / 1000.0 * sfreq)
        step_samp = max(1, int(self.step_ms / 1000.0 * sfreq))
        n_windows = (n_samples - window_samp) // step_samp

        times = np.zeros(n_windows)
        coefficients = np.zeros((self.n_modes, n_windows))
        phases_proj = np.zeros((self.n_modes, n_windows))

        # Eigenvectors for modes 1..n_modes (skip mode 0 = constant)
        V = self.eigvecs[:, 1:self.n_modes + 1]  # (n_electrodes, n_modes)

        # Zero out inactive electrodes in the eigenvectors
        V_masked = V.copy()
        V_masked[~active_mask] = 0
        # Renormalize
        for m in range(self.n_modes):
            norm = np.linalg.norm(V_masked[:, m])
            if norm > 1e-10:
                V_masked[:, m] /= norm

        t0 = time.time()
        report_every = max(1, n_windows // 10)

        for i in range(n_windows):
            start = i * step_samp
            center = start + window_samp // 2
            times[i] = center / sfreq

            # Circular mean phase per electrode in this window
            sin_mean = np.mean(np.sin(phases[:, start:start + window_samp]), axis=1)
            cos_mean = np.mean(np.cos(phases[:, start:start + window_samp]), axis=1)
            mean_phase = np.arctan2(sin_mean, cos_mean)

            # Complex field
            z_real = np.cos(mean_phase)
            z_imag = np.sin(mean_phase)

            # Project onto each eigenmode
            for m in range(self.n_modes):
                proj_r = np.dot(z_real, V_masked[:, m])
                proj_i = np.dot(z_imag, V_masked[:, m])
                coefficients[m, i] = np.sqrt(proj_r**2 + proj_i**2)
                phases_proj[m, i] = np.arctan2(proj_i, proj_r)

            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    [{label}] {100*(i+1)/n_windows:.0f}% "
                      f"({rate:.0f} win/s)")

        elapsed = time.time() - t0
        print(f"  [{label}] {n_windows} windows in {elapsed:.1f}s")
        return times, coefficients, phases_proj

    def compute_dwell_times_per_mode(self, phases_proj):
        """
        For each eigenmode, compute dwell times from the phase trajectory.
        A "dwell" is consecutive windows where phase changes < threshold.
        """
        all_dwells = []
        for m in range(self.n_modes):
            phase_traj = phases_proj[m]
            diffs = np.abs(np.diff(phase_traj))
            # Handle circular wrap
            diffs = np.minimum(diffs, 2 * np.pi - diffs)

            # Adaptive threshold: fraction of std
            threshold = self.stability_threshold * np.std(diffs)
            threshold = max(threshold, 0.05)  # minimum threshold

            is_stable = diffs < threshold
            dwells = []
            current_run = 0
            for stable in is_stable:
                if stable:
                    current_run += 1
                else:
                    if current_run > 0:
                        dwells.append(current_run)
                    current_run = 0
            if current_run > 0:
                dwells.append(current_run)

            dwell_ms = np.array(dwells) * self.step_ms if dwells else np.array([self.step_ms])
            all_dwells.append(dwell_ms)

        return all_dwells


# ═══════════════════════════════════════════════════════════════
# SINGLE SUBJECT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_single_subject(subject_dir, band='alpha', output_dir=None):
    """
    Analyze one subject: eyes-open (R01) vs eyes-closed (R02).
    """
    if output_dir is None:
        output_dir = subject_dir

    # Find baseline runs
    r01 = glob.glob(os.path.join(subject_dir, '*R01*.edf'))
    r02 = glob.glob(os.path.join(subject_dir, '*R02*.edf'))

    if not r01 or not r02:
        print(f"  Missing baseline runs in {subject_dir}")
        return None

    analyzer = EigenmodeMetastabilityAnalyzer(band=band, n_eigenmodes=8)

    # Eyes open
    phases_eo, mask_eo, sfreq, n_samp = analyzer.load_edf(r01[0])
    t_eo, coeff_eo, ph_eo = analyzer.compute_eigenmode_timeseries(
        phases_eo, mask_eo, sfreq, n_samp, label="EyesOpen")
    dwells_eo = analyzer.compute_dwell_times_per_mode(ph_eo)

    # Eyes closed
    phases_ec, mask_ec, sfreq2, n_samp2 = analyzer.load_edf(r02[0])
    t_ec, coeff_ec, ph_ec = analyzer.compute_eigenmode_timeseries(
        phases_ec, mask_ec, sfreq2, n_samp2, label="EyesClosed")
    dwells_ec = analyzer.compute_dwell_times_per_mode(ph_ec)

    # Pink noise control
    np.random.seed(42)
    phases_ctrl = analyzer.generate_pink_noise_phases(
        len(analyzer.e_names), n_samp, sfreq)
    mask_ctrl = mask_eo.copy()
    t_ctrl, coeff_ctrl, ph_ctrl = analyzer.compute_eigenmode_timeseries(
        phases_ctrl, mask_ctrl, sfreq, n_samp, label="Control")
    dwells_ctrl = analyzer.compute_dwell_times_per_mode(ph_ctrl)

    return {
        'analyzer': analyzer,
        'coeff_eo': coeff_eo, 'coeff_ec': coeff_ec, 'coeff_ctrl': coeff_ctrl,
        'ph_eo': ph_eo, 'ph_ec': ph_ec, 'ph_ctrl': ph_ctrl,
        'dwells_eo': dwells_eo, 'dwells_ec': dwells_ec, 'dwells_ctrl': dwells_ctrl,
        't_eo': t_eo, 't_ec': t_ec,
        'sfreq': sfreq,
        'eigvals': analyzer.eigvals,
        'eigvecs': analyzer.eigvecs,
    }


# ═══════════════════════════════════════════════════════════════
# MULTI-SUBJECT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_multi_subject(data_dir, band='alpha', max_subjects=20,
                          output_dir=None):
    """
    Analyze multiple subjects and aggregate results.
    """
    if output_dir is None:
        output_dir = data_dir

    # Find subject directories
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'S[0-9][0-9][0-9]')))
    if not subject_dirs:
        # Try without leading S
        subject_dirs = sorted(glob.glob(os.path.join(data_dir, '[0-9][0-9][0-9]')))
    if not subject_dirs:
        print(f"No subject directories found in {data_dir}")
        return

    subject_dirs = subject_dirs[:max_subjects]
    print(f"Found {len(subject_dirs)} subjects (analyzing {len(subject_dirs)})")

    n_modes = 8
    # Accumulators
    all_mean_dwells_eo = np.zeros((len(subject_dirs), n_modes))
    all_mean_dwells_ec = np.zeros((len(subject_dirs), n_modes))
    all_mean_dwells_ctrl = np.zeros((len(subject_dirs), n_modes))
    all_median_dwells_eo = np.zeros((len(subject_dirs), n_modes))
    all_median_dwells_ec = np.zeros((len(subject_dirs), n_modes))
    all_median_dwells_ctrl = np.zeros((len(subject_dirs), n_modes))
    all_cv_eo = np.zeros((len(subject_dirs), n_modes))
    all_cv_ec = np.zeros((len(subject_dirs), n_modes))
    all_cv_ctrl = np.zeros((len(subject_dirs), n_modes))
    all_kurt_eo = np.zeros((len(subject_dirs), n_modes))
    all_kurt_ec = np.zeros((len(subject_dirs), n_modes))

    # Per-mode aggregated dwells for KS tests
    agg_dwells_eo = [[] for _ in range(n_modes)]
    agg_dwells_ec = [[] for _ in range(n_modes)]
    agg_dwells_ctrl = [[] for _ in range(n_modes)]

    valid_subjects = 0

    for si, sdir in enumerate(subject_dirs):
        sname = os.path.basename(sdir)
        print(f"\n{'='*50}")
        print(f"Subject {sname} ({si+1}/{len(subject_dirs)})")
        print(f"{'='*50}")

        try:
            result = analyze_single_subject(sdir, band=band)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if result is None:
            continue

        for m in range(n_modes):
            d_eo = result['dwells_eo'][m]
            d_ec = result['dwells_ec'][m]
            d_ctrl = result['dwells_ctrl'][m]

            all_mean_dwells_eo[si, m] = np.mean(d_eo)
            all_mean_dwells_ec[si, m] = np.mean(d_ec)
            all_mean_dwells_ctrl[si, m] = np.mean(d_ctrl)
            all_median_dwells_eo[si, m] = np.median(d_eo)
            all_median_dwells_ec[si, m] = np.median(d_ec)
            all_median_dwells_ctrl[si, m] = np.median(d_ctrl)

            all_cv_eo[si, m] = np.std(d_eo) / (np.mean(d_eo) + 1e-9)
            all_cv_ec[si, m] = np.std(d_ec) / (np.mean(d_ec) + 1e-9)
            all_cv_ctrl[si, m] = np.std(d_ctrl) / (np.mean(d_ctrl) + 1e-9)

            if len(d_eo) > 4:
                all_kurt_eo[si, m] = stats.kurtosis(d_eo)
            if len(d_ec) > 4:
                all_kurt_ec[si, m] = stats.kurtosis(d_ec)

            agg_dwells_eo[m].extend(d_eo.tolist())
            agg_dwells_ec[m].extend(d_ec.tolist())
            agg_dwells_ctrl[m].extend(d_ctrl.tolist())

        valid_subjects += 1

    if valid_subjects < 2:
        print("Not enough valid subjects for group analysis")
        return

    # Trim to valid
    all_mean_dwells_eo = all_mean_dwells_eo[:valid_subjects]
    all_mean_dwells_ec = all_mean_dwells_ec[:valid_subjects]
    all_mean_dwells_ctrl = all_mean_dwells_ctrl[:valid_subjects]
    all_cv_eo = all_cv_eo[:valid_subjects]
    all_cv_ec = all_cv_ec[:valid_subjects]
    all_cv_ctrl = all_cv_ctrl[:valid_subjects]
    all_kurt_eo = all_kurt_eo[:valid_subjects]
    all_kurt_ec = all_kurt_ec[:valid_subjects]

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"GROUP RESULTS: {valid_subjects} subjects, {band} band")
    print(f"{'='*70}")

    print(f"\n{'Mode':>6} | {'EO Dwell':>10} {'EC Dwell':>10} {'Ctrl':>10} | "
          f"{'EO-EC t':>8} {'EO-EC p':>10} | {'EO-Ctrl t':>9} {'EO-Ctrl p':>10} | "
          f"{'EO CV':>6} {'EC CV':>6}")
    print("─" * 110)

    for m in range(n_modes):
        eo_m = all_mean_dwells_eo[:, m]
        ec_m = all_mean_dwells_ec[:, m]
        ctrl_m = all_mean_dwells_ctrl[:, m]

        # Paired t-test: eyes open vs eyes closed
        t_eo_ec, p_eo_ec = stats.ttest_rel(eo_m, ec_m)
        # Independent t-test: eyes open vs control
        t_eo_ctrl, p_eo_ctrl = stats.ttest_ind(eo_m, ctrl_m)

        sig_ec = "***" if p_eo_ec < 0.001 else "**" if p_eo_ec < 0.01 else "*" if p_eo_ec < 0.05 else ""
        sig_ctrl = "***" if p_eo_ctrl < 0.001 else "**" if p_eo_ctrl < 0.01 else "*" if p_eo_ctrl < 0.05 else ""

        print(f"  M{m+1:>3} | {np.mean(eo_m):>8.1f}ms {np.mean(ec_m):>8.1f}ms "
              f"{np.mean(ctrl_m):>8.1f}ms | {t_eo_ec:>+7.2f} {p_eo_ec:>9.2e}{sig_ec:>3} | "
              f"{t_eo_ctrl:>+8.2f} {p_eo_ctrl:>9.2e}{sig_ctrl:>3} | "
              f"{np.mean(all_cv_eo[:, m]):>5.2f} {np.mean(all_cv_ec[:, m]):>5.2f}")

    # ── Aggregated KS tests ──
    print(f"\nAggregated KS tests (all subjects pooled):")
    print(f"{'Mode':>6} | {'EO vs Ctrl KS':>14} {'p':>12} | {'EC vs Ctrl KS':>14} {'p':>12} | {'EO vs EC KS':>12} {'p':>12}")
    print("─" * 95)
    for m in range(n_modes):
        eo_all = np.array(agg_dwells_eo[m])
        ec_all = np.array(agg_dwells_ec[m])
        ctrl_all = np.array(agg_dwells_ctrl[m])

        ks1, p1 = stats.ks_2samp(eo_all, ctrl_all) if len(eo_all) > 5 and len(ctrl_all) > 5 else (0, 1)
        ks2, p2 = stats.ks_2samp(ec_all, ctrl_all) if len(ec_all) > 5 and len(ctrl_all) > 5 else (0, 1)
        ks3, p3 = stats.ks_2samp(eo_all, ec_all) if len(eo_all) > 5 and len(ec_all) > 5 else (0, 1)

        print(f"  M{m+1:>3} | {ks1:>12.4f} {p1:>12.2e} | {ks2:>12.4f} {p2:>12.2e} | {ks3:>12.4f} {p3:>12.2e}")

    # ── Visualization ──
    print("\nGenerating figures...")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Eigenmode Metastability: {valid_subjects} subjects, {band} band\n"
                 f"Graph Laplacian decomposition of EEG phase fields",
                 fontsize=14, fontweight='bold', color='white')

    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Mean dwell per eigenmode ──
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(n_modes)
    w = 0.25
    ax.bar(x - w, np.mean(all_mean_dwells_eo, axis=0), w, color='#2196F3',
           label='Eyes Open', alpha=0.8, yerr=np.std(all_mean_dwells_eo, axis=0)/np.sqrt(valid_subjects))
    ax.bar(x, np.mean(all_mean_dwells_ec, axis=0), w, color='#FF9800',
           label='Eyes Closed', alpha=0.8, yerr=np.std(all_mean_dwells_ec, axis=0)/np.sqrt(valid_subjects))
    ax.bar(x + w, np.mean(all_mean_dwells_ctrl, axis=0), w, color='#9E9E9E',
           label='Pink Noise', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i+1}' for i in range(n_modes)], fontsize=8)
    ax.set_ylabel('Mean Dwell (ms)')
    ax.set_title('Mean Dwell Time per Eigenmode', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_facecolor('#1a1a2e')

    # ── Panel 2: CV per eigenmode ──
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - w, np.mean(all_cv_eo, axis=0), w, color='#2196F3',
           label='Eyes Open', alpha=0.8)
    ax.bar(x, np.mean(all_cv_ec, axis=0), w, color='#FF9800',
           label='Eyes Closed', alpha=0.8)
    ax.bar(x + w, np.mean(all_cv_ctrl, axis=0), w, color='#9E9E9E',
           label='Pink Noise', alpha=0.6)
    ax.axhline(1.0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i+1}' for i in range(n_modes)], fontsize=8)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('CV of Dwell Times (>1 = heavy-tailed)', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_facecolor('#1a1a2e')

    # ── Panel 3: Kurtosis ──
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x - 0.15, np.mean(all_kurt_eo, axis=0), 0.3, color='#2196F3',
           label='Eyes Open', alpha=0.8)
    ax.bar(x + 0.15, np.mean(all_kurt_ec, axis=0), 0.3, color='#FF9800',
           label='Eyes Closed', alpha=0.8)
    ax.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'M{i+1}' for i in range(n_modes)], fontsize=8)
    ax.set_ylabel('Excess Kurtosis')
    ax.set_title('Distribution Tail Weight', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_facecolor('#1a1a2e')

    # ── Panel 4-6: Survival functions for modes 1, 2, 4 ──
    for panel_idx, mode_idx in enumerate([0, 1, 3]):
        ax = fig.add_subplot(gs[1, panel_idx])
        eo_all = np.array(agg_dwells_eo[mode_idx])
        ec_all = np.array(agg_dwells_ec[mode_idx])
        ctrl_all = np.array(agg_dwells_ctrl[mode_idx])

        for data, color, label in [(eo_all, '#2196F3', 'EO'),
                                    (ec_all, '#FF9800', 'EC'),
                                    (ctrl_all, '#9E9E9E', 'Ctrl')]:
            if len(data) > 5:
                sorted_d = np.sort(data)[::-1]
                ccdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                ax.loglog(sorted_d, ccdf, '.', color=color, markersize=2,
                          alpha=0.7, label=label)

        ax.set_xlabel('Dwell Time (ms)')
        ax.set_ylabel('P(X > x)')
        ax.set_title(f'Survival: Eigenmode {mode_idx+1}\n'
                     f'(λ={result["eigvals"][mode_idx+1]:.2f})', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15, which='both')
        ax.set_facecolor('#1a1a2e')

    # ── Panel 7: Eigenmode spatial patterns ──
    ax = fig.add_subplot(gs[2, 0])
    coords = result['analyzer'].e_coords
    eigvecs = result['eigvecs']

    # Show modes 1-4 as colored scatter
    for mi, (color, marker) in enumerate(zip(['#2196F3', '#FF9800', '#4CAF50', '#F44336'],
                                              ['o', 's', '^', 'D'])):
        if mi >= n_modes:
            break
        v = eigvecs[:, mi + 1]
        # Scale for visibility
        sizes = np.abs(v) / (np.max(np.abs(v)) + 1e-9) * 100 + 10
        colors_arr = np.where(v > 0, 1.0, -1.0)
        ax.scatter(coords[:, 0], coords[:, 1], c=v, s=sizes,
                   cmap='RdBu_r', alpha=0.3, edgecolors='none')

    # Just show mode 2 clearly
    v2 = eigvecs[:, 1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=v2, s=60,
                         cmap='RdBu_r', alpha=0.9, edgecolors='white',
                         linewidth=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Eigenmode 1 (Anterior-Posterior)', fontsize=9)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    ax.set_facecolor('#1a1a2e')

    # ── Panel 8: EO vs EC effect sizes ──
    ax = fig.add_subplot(gs[2, 1])
    effect_sizes = []
    p_vals = []
    for m in range(n_modes):
        eo_m = all_mean_dwells_eo[:, m]
        ec_m = all_mean_dwells_ec[:, m]
        pooled_std = np.sqrt((np.std(eo_m)**2 + np.std(ec_m)**2) / 2)
        d = (np.mean(ec_m) - np.mean(eo_m)) / (pooled_std + 1e-9)
        _, p = stats.ttest_rel(eo_m, ec_m)
        effect_sizes.append(d)
        p_vals.append(p)

    colors_bar = ['#4CAF50' if p < 0.05 else '#F44336' for p in p_vals]
    ax.barh(x, effect_sizes, color=colors_bar, alpha=0.8)
    ax.axvline(0, color='white', linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels([f'M{i+1}' for i in range(n_modes)], fontsize=8)
    ax.set_xlabel("Cohen's d (EC - EO)")
    ax.set_title('Eyes Closed vs Open Effect Size\n(green=sig, >0=EC longer)', fontsize=9)
    ax.set_facecolor('#1a1a2e')

    # ── Panel 9: Eigenvalue spectrum ──
    ax = fig.add_subplot(gs[2, 2])
    eigvals = result['eigvals']
    ax.plot(range(1, len(eigvals)), eigvals[1:], 'o-', color='#2196F3',
            markersize=3, linewidth=1)
    for mi in range(min(n_modes, len(eigvals) - 1)):
        ax.axvline(mi + 1, color='white', alpha=0.1, linewidth=0.5)
    ax.set_xlabel('Eigenmode Index')
    ax.set_ylabel('Eigenvalue (λ)')
    ax.set_title('Graph Laplacian Spectrum\n(low λ = slow/global, high λ = fast/local)',
                 fontsize=9)
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, min(30, len(eigvals)))

    out_path = os.path.join(output_dir, f'eigenmode_metastability_{band}_{valid_subjects}subj.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\nSaved: {out_path}")
    plt.close()

    return {
        'n_subjects': valid_subjects,
        'mean_dwells_eo': all_mean_dwells_eo,
        'mean_dwells_ec': all_mean_dwells_ec,
        'mean_dwells_ctrl': all_mean_dwells_ctrl,
        'agg_dwells_eo': agg_dwells_eo,
        'agg_dwells_ec': agg_dwells_ec,
        'agg_dwells_ctrl': agg_dwells_ctrl,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eigenmode Metastability Analyzer - "
                    "Graph Laplacian decomposition of EEG phase fields"
    )
    parser.add_argument('data_path',
                        help='Path to eegmmidb directory or single subject directory')
    parser.add_argument('--band', default='alpha',
                        choices=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        help='Frequency band (default: alpha)')
    parser.add_argument('--subjects', type=int, default=20,
                        help='Max subjects for multi-subject analysis (default: 20)')
    parser.add_argument('--window', type=int, default=250,
                        help='Window size in ms (default: 250)')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size in ms (default: 10)')
    parser.add_argument('--output', default=None,
                        help='Output directory')
    parser.add_argument('--all', action='store_true',
                        help='Run all bands')
    parser.add_argument('--single', action='store_true',
                        help='Analyze single subject only')

    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output or (os.path.dirname(os.path.abspath(data_path))
                                  if os.path.isfile(data_path)
                                  else os.path.abspath(data_path))

    if not os.path.isdir(output_dir):
        output_dir = '.'

    print("╔══════════════════════════════════════════════════╗")
    print("║   EIGENMODE METASTABILITY ANALYZER               ║")
    print("║   Graph Laplacian × Holographic Phase Fields     ║")
    print("║   Pribram → Wang et al. → Baker & Cariani        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    bands_to_run = list(BANDS.keys()) if args.all else [args.band]

    for band in bands_to_run:
        print(f"\n{'#'*60}")
        print(f"# BAND: {band.upper()}")
        print(f"{'#'*60}")

        if args.single or os.path.isfile(data_path):
            # Single subject
            sdir = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
            result = analyze_single_subject(sdir, band=band, output_dir=output_dir)
        else:
            # Multi-subject
            # Check if data_path contains S001/ or if we need to go deeper
            if os.path.isdir(os.path.join(data_path, 'S001')):
                base = data_path
            elif os.path.isdir(os.path.join(data_path, '1.0.0', 'S001')):
                base = os.path.join(data_path, '1.0.0')
            elif os.path.isdir(os.path.join(data_path, 'files', 'S001')):
                base = os.path.join(data_path, 'files')
            else:
                base = data_path

            result = analyze_multi_subject(
                base, band=band,
                max_subjects=args.subjects,
                output_dir=output_dir
            )

    print("\nDone.")
