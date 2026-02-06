# Φ-Dwell: Eigenmode Phase-Field Metastability Suite

**Measuring how the brain moves through its own geometry.**

Φ-Dwell is a suite of tools that decomposes scalp EEG into spatial eigenmodes of the electrode graph Laplacian and measures the temporal dynamics of coherent phase-field configurations.

Rather than treating brain activity as continuous oscillation, Φ-Dwell shows that the brain **dwells in discrete spatial configurations**, transitions between them with **heavy-tailed statistics**, and **expands its configuration vocabulary during cognition**.

---

## What This Measures (and What It Doesn’t)

### What Φ-Dwell measures
Traditional EEG analysis focuses on **power** — how strong a frequency band is.  
Φ-Dwell measures something orthogonal:

- How long a spatial phase configuration remains coherent (**dwell time**)
- How configurations transition between one another (**transition grammar**)
- The statistical regime of these transitions (exponential, critical, bursty, clocklike)

### Core operation
At each time step:

- 64 electrodes × 5 frequency bands
- Project instantaneous phase fields onto spatial eigenmodes of the electrode geometry
- Result: a **40-dimensional state vector** (8 eigenmodes × 5 bands)

This vector traces a trajectory through **brain configuration space**.  
Its dwell times and transitions encode the brain’s dynamical state.

### What it does *not* measure
- Individual neurons or spikes  
- Thought content  
- Single-cell activity  

All measurements are **macroscopic**, volume-conducted signatures of large-scale neural coordination.

---

## Key Findings

### 1. Five-Band Dwell Hierarchy (11 subjects)

Mean dwell times for Mode 1 (anterior–posterior eigenmode), eyes open:

| Band  | Mean Dwell | CV   | Regime              |
|------|------------|------|---------------------|
| Delta | 151 ms     | ~1.0 | Near-exponential    |
| Theta | 27 ms      | ~1.4 | Critical (power-law)|
| Alpha | 16 ms      | ~1.3 | Critical (power-law)|
| Beta  | 13 ms      | ~1.3 | Bursty              |
| Gamma | 12 ms      | ~0.5 | Clocklike           |

---

### 2. Alpha–Theta Double Dissociation (11 subjects)

| Band  | Eyes Open | Eyes Closed | Direction | p-value |
|------|-----------|-------------|-----------|--------|
| Theta | 26.5 ms  | 23.5 ms     | EO > EC   | 0.008  |
| Alpha | 16.3 ms  | 21.1 ms     | EC > EO   | 0.068  |

---

### 3. Universal Eigenmode Persistence (20 subjects)

| Band  | Self-Transition | CV Across Subjects |
|------|-----------------|-------------------|
| Delta | 0.898 ± 0.004 | 0.5% |
| Theta | 0.849 ± 0.004 | 0.5% |
| Gamma | 0.815 ± 0.005 | 0.6% |
| Alpha | 0.809 ± 0.006 | 0.7% |
| Beta  | 0.798 ± 0.005 | 0.7% |

---

### 4. Task-Dependent Vocabulary Expansion (5 subjects)

Motor imagery vs rest:

| Subject | Rest | Task | Ratio | Task-Only | Jaccard |
|-------|------|------|-------|----------|---------|
| S001 | 822 | 1,885 | 2.3× | 1,240 | 0.31 |
| S002 | 782 | 1,535 | 2.0× | 944 | 0.34 |
| S003 | 840 | 1,899 | 2.3× | 1,257 | 0.31 |
| S004 | 941 | 2,640 | 2.8× | 1,965 | 0.23 |
| S005 | 1,628 | 3,467 | 2.1× | 2,386 | 0.27 |

---

## Tool Suite

See repository scripts:
- eigenmode_metastability.py
- phidwell_deep_analyzer.py
- phidwell_cross_subjects.py
- phidwell_grammar_decoder.py
- phidwell_macroscope_standalone.py
- brain_grammar.py

---

## Dataset

Validated on the PhysioNet EEG Motor Movement/Imagery Dataset:

- 109 subjects
- 64 channels (10–10 system)
- 160 Hz sampling

```bash
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

---

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib
- mne
- scikit-learn

---

## Citation

If you use Φ-Dwell in your research:

> Luode, A. & Claude (Anthropic). (2025).  
> **Φ-Dwell: Eigenmode Phase-Field Metastability Suite**  
> https://github.com/anttiluode/BrainMetastabilityAnalyzerTool/

---

## License
MIT

---

## Acknowledgments

Developed through collaborative research sessions between  
**Antti Luode (PerceptionLab)** and **Claude (Anthropic)**, February 2025.
