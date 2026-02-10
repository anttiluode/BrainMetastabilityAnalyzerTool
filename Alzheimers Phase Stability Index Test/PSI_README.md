# Phase-Stability Index (PSI) — Alzheimer's Analyzer Extension

## What it does

`phidwell_alzheimers_with_psi.py` extends the standard Φ-Dwell Alzheimer's analyzer with a new metric: the **Phase-Stability Index (PSI)**. For each time window of EEG data, PSI measures how many iterations of Gerchberg-Saxton phase recovery are needed for the eigenmode power distribution to converge to a stable spatial pattern.

The idea: a brain state's eigenmode power vector defines a radial magnitude profile. Some profiles strongly constrain what spatial patterns are possible (converge fast = high phase-stability). Others are ambiguous (converge slow = low phase-stability). PSI measures this "phase recoverability" — how internally coherent each moment of brain dynamics is.

### How it works

1. Takes the same eigenmode coefficient matrix already computed by Φ-Dwell (5 bands × timesteps × 6 modes)
2. For each time window, maps the eigenmode powers onto a radial magnitude profile on a small (32×32) canvas
3. Runs iterative Gerchberg-Saxton phase recovery with temporal memory between windows (inertia=0.85)
4. Records how many iterations (out of 20 max) until phase change drops below threshold (0.05 rad)
5. Aggregates across windows: mean iterations, standard deviation, per-band mode entropy, band gradient

## Results on ds004504

Run on 88 subjects (36 AD, 23 FTD, 29 CN):

| Metric | CN | AD | FTD | KW p | AD vs CN p |
|---|---|---|---|---|---|
| **PSI Mean Iters** | **5.00** | **4.55** | **4.52** | **0.018** | **0.006** |
| PSI Std | 2.68 | 2.49 | 2.38 | 0.101 | 0.068 |
| PSI Band Gradient | 0.046 | 0.046 | 0.043 | 0.645 | ns |
| PSI Residual | 0.004 | 0.003 | 0.003 | 0.078 | 0.121 |
| PSI % Converged | 0.996 | 0.995 | 0.997 | 0.667 | ns |

**PSI Mean Iterations** is the strongest single metric for AD vs CN separation (Mann-Whitney p=0.006, effect size r=0.401), outperforming vocabulary size (p=0.021), entropy (p=0.016), and mean CV (p=0.010).

### Interpretation

- **Healthy controls converge in ~5 iterations**: enough eigenmode complexity to require active phase recovery, but structured enough to find a stable solution
- **AD and FTD converge in ~4.5 iterations**: eigenmode configurations are simpler, more repetitive — the phase recovery "resolves" them too easily because the underlying dynamics have collapsed to a lower-dimensional manifold
- This is consistent with the other Φ-Dwell findings: AD shows lower Zipf α (0.612 vs 0.681), lower top-5 concentration (0.124 vs 0.154), and lower CV (0.945 vs 0.983) — all indicating loss of structured dynamics

### Important caveats

- PSI does **not** significantly correlate with MMSE (ρ=0.165, p=0.125). It separates groups but doesn't track within-group severity. This suggests it captures a threshold structural property rather than a continuous decline measure.
- PSI Band Gradient did not separate groups (p=0.645). The hypothesis that frequency-band hierarchy would be disrupted was not supported at this resolution.
- One outlier: sub-017 (MMSE=6, most severe AD) showed PSI=6.9 — much higher than the AD mean. This could indicate a U-shaped curve where severe AD dynamics become chaotic rather than simplified, but N=1 is not evidence.
- No multiple comparison correction applied across the 15 metrics tested. PSI's p=0.018 would survive Bonferroni correction at the 15-test level (threshold 0.003) only for the pairwise AD vs CN test (p=0.006).
- Single dataset, single run, no cross-validation.

### Dissociation from existing metrics

The most interesting finding is that PSI and the CV/Zipf metrics capture different aspects of brain dynamics:

- **PSI** separates groups (p=0.018) but does not correlate with MMSE (p=0.125)
- **Zipf α** and **Criticality Fraction** correlate with MMSE (p=0.043, 0.032) but are weaker on group separation

PSI appears to measure a structural property (is the eigenmode configuration space collapsed?) while CV and Zipf measure dynamic properties (how does the brain navigate its available space?). If this dissociation replicates, it suggests two independent axes of neurodegeneration.

## Usage

Identical to the standard analyzer:

```bash
python phidwell_alzheimers_with_psi.py "path/to/ds004504/"
```

Adds ~2-3 seconds per subject for PSI computation (500 windows × 20 GS iterations on 32×32 canvas).

## Origin

The PSI metric emerged from work on iterated radial↔Cartesian projection operators in the Perception Laboratory visual programming environment. The core observation: when EEG eigenmode power vectors are treated as radial magnitude profiles and subjected to iterative phase recovery, the convergence rate varies systematically with brain state. The operator's convergence dynamics provide a novel probe of eigenmode configuration coherence that is mathematically independent of the existing Φ-Dwell vocabulary and dwell-time metrics.
