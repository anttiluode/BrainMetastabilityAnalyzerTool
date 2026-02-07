# Φ-Dwell: Eigenmode Phase-Field Metastability Analysis of Scalp EEG

**Measuring how the brain moves through its own geometry.**

Φ-Dwell decomposes scalp EEG into spatial eigenmodes of the electrode graph Laplacian and measures the temporal dynamics of coherent phase-field configurations. Rather than asking "how much alpha power is there?", Φ-Dwell asks "how long does the spatial pattern hold its shape, what vocabulary of configurations does the brain use, and does that vocabulary change in disease?"

The suite has been validated on 109 healthy subjects (PhysioNet Motor Movement/Imagery Dataset) and 88 clinical subjects (OpenNeuro ds004504: 36 Alzheimer's, 23 frontotemporal dementia, 29 age-matched controls).

---

## Key Findings

### 1. Five-Band Dwell Hierarchy

Each frequency band operates in a qualitatively different dynamical regime. Mean dwell times for Mode 1 (anterior-posterior), eyes open, 11 subjects:

| Band  | Mean Dwell | CV   | Regime           |
|-------|-----------|------|------------------|
| Delta | 151 ms    | ~1.0 | Near-exponential |
| Theta | 27 ms     | ~1.4 | Critical         |
| Alpha | 16 ms     | ~1.3 | Critical         |
| Beta  | 13 ms     | ~1.3 | Bursty           |
| Gamma | 12 ms     | ~0.5 | Clocklike        |

### 2. Alpha-Theta Double Dissociation

Theta dwells longer eyes-open (p=0.008). Alpha dwells longer eyes-closed (p=0.068). Opposite directions in adjacent bands rule out generic artifacts.

### 3. Universal Eigenmode Persistence

Self-transition rates across 20 subjects show CV < 1%. The hierarchy δ > θ > γ > α > β is preserved in every subject. 83% of attractors are critical (CV > 1.0).

### 4. Task Doubles the Vocabulary

Motor imagery expands the eigenmode configuration vocabulary from ~800 to ~2,000 words (5/5 subjects). Two-thirds of task words never appear at rest.

### 5. Rest and Task Follow Different Grammars

Cross-condition bigram perplexity is 10-25× higher than within-condition (5/5 subjects, p=0.049). The brain during task doesn't just use more words — it follows different sequential transition rules.

### 6. Eigenmode Vocabulary Detects Alzheimer's Disease

On OpenNeuro ds004504 (88 subjects), five Φ-Dwell metrics significantly separate AD and FTD from healthy controls:

| Metric | CN | AD | FTD | p-value |
|--------|----|----|-----|---------|
| Vocabulary Size | 953 | 1052 | 1078 | 0.035 |
| Shannon Entropy | 8.26 | 8.57 | 8.61 | 0.032 |
| Mean CV (Criticality) | 0.983 | 0.945 | 0.945 | 0.022 |
| Top-5 Concentration | 0.154 | 0.124 | 0.127 | 0.035 |
| Zipf α | 0.681 | 0.612 | 0.628 | 0.034 |

AD brains show *more* vocabulary with *less* structure — loss of organized dynamics, not loss of repertoire. The brain can no longer hold a pattern.

### 7. Dwell Gradient: Strongest Alzheimer's Biomarker

The β-Sieve analysis reveals the dwell gradient across the band hierarchy (δ→γ) as the single most discriminative metric (KW p=0.0015, MMSE correlation ρ=0.408, p=0.0001), driven by collapse of alpha-band eigenmode stability.

---

## Tool Suite

### Core Analysis

**`eigenmode_metastability.py`** — Foundation analyzer. Computes dwell-time distributions per eigenmode, per band, per condition. Compares eyes-open vs eyes-closed. Produces multi-panel figures with survival functions, effect sizes, and topographic maps.

```bash
python eigenmode_metastability.py "physionet.org/files/eegmmidb/1.0.0/" --band alpha --subjects 20
python eigenmode_metastability.py "physionet.org/files/eegmmidb/1.0.0/" --subjects 109 --all
```

**`phidwell_deep_analyzer.py`** — Configuration space explorer. Extracts the full attractor catalog, eigenmode transition grammar, cross-band coupling matrix, and regime classification. Saves JSON fingerprints.

```bash
python phidwell_deep_analyzer.py "physionet.org/files/eegmmidb/1.0.0/" --subjects 20
```

**`phidwell_cross_subjects.py`** — Population-level analysis. Loads fingerprint JSONs, identifies universals vs individual differences, performs hierarchical clustering.

```bash
python phidwell_cross_subjects.py "physionet.org/files/eegmmidb/1.0.0/phidwell_all_fingerprints.json"
```

### Vocabulary & Grammar

**`phidwell_grammar_decoder.py`** — Rest vs task vocabulary comparison. Tokenizes EEG into eigenmode "words," compares resting baseline to motor imagery, computes vocabulary expansion, entropy shifts, enriched/depleted words, and laterality measures.

```bash
python phidwell_grammar_decoder.py "physionet.org/files/eegmmidb/1.0.0/" --subjects 5
```

**`phidwell_perplexity.py`** — Cross-condition perplexity analyzer. Trains bigram models on rest and task eigenmode word sequences, then measures cross-perplexity. Quantifies the "distance of thought" — how different task grammar is from rest grammar. Includes masked prediction accuracy, surprisal timecourses, and Zipf analysis.

```bash
python phidwell_perplexity.py "physionet.org/files/eegmmidb/1.0.0/" --subjects 5
```

**`braingrammargemini.py`** — Standalone eigenmode vocabulary tokenizer. Converts continuous EEG into discrete eigenmode words and computes vocabulary statistics, bigram syntax, and entropy.

### Clinical Application

**`phidwell_alzheimers.py`** — Alzheimer's and FTD detection from resting EEG. Processes OpenNeuro ds004504 (19-channel, 10-20 system). For each of 88 subjects computes: vocabulary size, entropy, perplexity, self-transition rate, cross-band coupling, criticality (CV), Zipf exponent, and per-band metrics. Runs Kruskal-Wallis tests across AD/FTD/CN groups, pairwise Mann-Whitney with effect sizes, and MMSE correlations. Produces 12-panel diagnostic dashboard.

```bash
# Download the dataset
openneuro download --snapshot 1.0.8 ds004504 ds004504/

# Run analysis
python phidwell_alzheimers.py "path/to/ds004504/"
```

**`brain_viscosity.py`** — β-Sieve × Φ-Dwell brain viscosity analyzer. Applies the β-gradient concept from neural network grokking research to brain eigenmode dynamics. The frequency band hierarchy (δ→θ→α→β→γ) serves as a "depth" axis analogous to neural network layers. Computes roughness profiles, dwell gradients, and a combined viscosity index. Runs from existing Φ-Dwell JSON results (instant) or directly on raw EEG .set files.

```bash
# From existing results (fast reanalysis)
python brain_viscosity.py phidwell_alzheimer_results.json

# From raw EEG (full pipeline)
python brain_viscosity.py "path/to/ds004504/" --full
```

### Visualization

**`phidwell_macroscope_standalone.py`** — Real-time visualization dashboard with 7 panels: phase portrait, regime detector, dwell histogram, eigenmode bars, band power, and trajectory trail. Works with EDF files or synthetic demo mode.

```bash
python phidwell_macroscope_standalone.py "path/to/file.edf" --band alpha
python phidwell_macroscope_standalone.py --demo
```

---

## Method

At each time step, for N electrodes × 5 frequency bands:

1. Bandpass filter → Hilbert transform → instantaneous phase per channel
2. Project the phase field onto spatial eigenmodes of the electrode graph Laplacian
3. Track dominant eigenmode per band over time
4. Tokenize into discrete "words" (5-tuples of dominant mode per band)
5. Analyze: dwell times, vocabulary, transition grammar, perplexity, criticality

The graph Laplacian eigenmodes form a natural spatial basis ordered by spatial frequency:

| Mode | Pattern |
|------|---------|
| 1 | Anterior ↔ Posterior |
| 2 | Left ↔ Right |
| 3 | Center ↔ Periphery |
| 4 | Diagonal quadrants |
| 5-8 | Higher-order patterns |

This follows directly from spectral graph theory of brain networks (Wang, Owen, Mukherjee & Raj, 2017, *PLoS Computational Biology*). Where Raj's group applies the graph Laplacian to the structural connectome from diffusion MRI tractography, Φ-Dwell applies it to the electrode geometry and measures how the brain's *functional dynamics* move through eigenmode space over time.

---

## Datasets

**PhysioNet EEG Motor Movement/Imagery Dataset** — 109 subjects, 64 channels (10-10 system), 160 Hz. Eyes-open/closed baselines plus motor execution and imagery tasks.

```bash
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

**OpenNeuro ds004504** — 36 Alzheimer's, 23 FTD, 29 healthy controls. 19 channels (10-20 system), 500 Hz, resting-state eyes-closed. MMSE scores available for all 88 subjects. Published by Miltiadous et al. (2023).

```bash
npm install -g @openneuro/cli
openneuro download --snapshot 1.0.8 ds004504 ds004504/
```

---

## Results

The `Results/` folder contains output files from analyses including JSON data, PNG figures, and summary statistics.

`PAPER.md.pdf` contains the full technical writeup of the Φ-Dwell framework and findings.

---

## Theoretical Context

Φ-Dwell connects four research programs:

- **Spectral graph theory of brain networks** (Raj et al., 2012; Wang et al., 2017) — Laplacian eigenmodes as the natural basis for brain dynamics and disease progression
- **Baker & Cariani's oscillatory cascade** (2025) — Frequency-band hierarchy from fast to slow as depth of cognitive processing
- **Critical brain hypothesis** — Optimal computation at the edge of criticality, measured directly via dwell-time coefficient of variation
- **Phase transitions in learning** — The β-Sieve viscosity probe bridges neural network grokking dynamics with brain eigenmode organization

---

## Requirements

```
Python 3.10+
numpy, scipy, matplotlib, mne
Optional: scikit-learn, tabulate, networkx
```

For OpenNeuro dataset downloads:
```
npm install -g @openneuro/cli
```

---

## What This Is and What It Isn't

**What it measures:** Temporal persistence and sequential grammar of macroscopic phase-field configurations projected onto spatial eigenmodes. Volume-conducted signatures of large-scale neural coordination.

**What it does not measure:** Individual neurons, thought content, or single-cell activity.

**What's novel:** Eigenmode vocabulary size, eigenmode transition grammar perplexity, cross-band eigenmode coupling, dwell gradient across the band hierarchy, and their application as neurodegeneration biomarkers.

**What needs more work:** AD/FTD differential diagnosis did not separate at 19 channels (would need 64+). Coupling metrics need higher spatial resolution. Task perplexity results are from 5 subjects. All findings are from public datasets and await independent replication.

---

## Citation

> Luode, A. & Claude (Anthropic). (2025).
> **Φ-Dwell: Eigenmode Phase-Field Metastability Analysis of Scalp EEG.**
> https://github.com/anttiluode/BrainMetastabilityAnalyzerTool/

---

## License

MIT
