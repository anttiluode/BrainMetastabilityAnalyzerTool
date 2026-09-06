# BrainMetastabilityAnalyzerTool — 2026 Reality Audit

Status: **discovery-stage research code**. The repository contains a defensible EEG state-space representation, but its historical Alzheimer/FTD language overstates what the existing experiments establish.

This document separates **measurement**, **discovery**, **internal robustness**, and **external validation**.

## 1. Core measurement that survives

The defensible core is:

1. band-pass scalp EEG;
2. Hilbert phase per electrode;
3. project the complex phase field onto graph-Laplacian eigenvectors built from electrode geometry;
4. choose the dominant spatial mode per frequency band;
5. measure persistence, transitions, and multi-band state occupancy.

These are best described as **spatial sensor-phase modes**. The graph is constructed from sensor coordinates, not from diffusion-MRI tracts or a cortical structural connectome.

No claim about holography, thought content, consciousness, or a literal brain grammar is required for the measurement to be useful.

## 2. Historical claims reclassified

### KEEP

- Spatial sensor-phase mode projection.
- Per-band dominant-mode dwell times.
- Transition statistics.
- Multi-band state words as an exploratory discretization.
- `dwell_gradient` as a single candidate for independent replication.

### EXPLORATORY / QUARANTINE

- vocabulary size;
- entropy;
- Zipf exponent;
- top-5 concentration;
- training perplexity;
- PSI / Gerchberg–Saxton convergence;
- the `viscosity` interpretation.

Several vocabulary metrics summarize the same state-count distribution and should not be presented as independent biomarkers.

### REPAIR / RETIRE

**Categorical Pearson coupling.** Dominant mode IDs are labels. Pearson correlation changes under arbitrary relabeling and is therefore not a sound categorical coupling measure. Use a label-invariant measure such as mutual information or continuous coefficient vectors.

**Criticality from CV > 1.** Dwell-time coefficient of variation is a legitimate statistic. A threshold of `CV > 1` does not demonstrate a critical phase transition. Use `dwell_cv` unless stronger criticality tests are added.

**In-sample bigram perplexity.** The historical Alzheimer analyzer builds and scores the bigram model on the same sequence. This is descriptive training perplexity, not predictive generalization.

## 3. The task-vocabulary result is confounded

The historical PhysioNet task analysis compared unequal observation amounts:

- REST concatenated two baseline recordings;
- TASK concatenated six longer motor-imagery recordings;
- whole motor-imagery files were labeled TASK despite alternating `T0` rest and `T1/T2` task blocks.

Vocabulary size naturally grows with observation time. Therefore the statement that “task doubles the vocabulary” is not currently established. A clean rerun must use equal-duration event-conditioned T0 versus T1/T2 samples.

## 4. Alzheimer discovery dataset

Historical Alzheimer's work used OpenNeuro `ds004504`:

- 36 Alzheimer's disease (AD)
- 23 frontotemporal dementia (FTD)
- 29 cognitively normal controls (CN)
- 19 scalp EEG channels
- resting eyes closed

The strongest later candidate was derived after the original results had already been inspected:

```text
dwell_gradient = slope(log(1 + mean_dwell_band))
                 across delta → theta → alpha → beta → gamma
```

Historical discovery receipt:

```text
Kruskal-Wallis across CN/AD/FTD    p = 0.0015
AD vs CN                           p = 0.0003
pooled MMSE Spearman               rho = 0.408, p = 0.0001
```

These are discovery statistics, not confirmation.

## 5. 2026 internal spectral-slowing audit — completed

The audit harness was written before this output was inspected. It processed all 88 subjects with zero failures using the frozen dwell-gradient definition and three ordinary spectral baselines.

### AD versus controls

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| `dwell_gradient` | -0.4469 | -0.4164 | 0.000277 |
| alpha relative power | 0.0481 | 0.0755 | 0.004101 |
| theta / alpha ratio | 2.6013 | 1.8297 | 0.000993 |
| peak alpha frequency | 7.479 Hz | 8.664 Hz | 0.000124 |

Ordinary spectral slowing is therefore plainly present in this cohort. Peak alpha frequency is at least as striking a univariate group marker as dwell gradient.

### Severity sanity check

The historical pooled MMSE association does not become a clean within-disease severity effect:

```text
within AD:  dwell_gradient vs MMSE  rho = 0.2154, p = 0.2072, n = 36
within FTD: dwell_gradient vs MMSE  rho = 0.1210, p = 0.5824, n = 23
```

This substantially weakens the old “tracks severity” interpretation. The pooled MMSE relationship was entangled with diagnostic-group separation.

### Internal repeated subject-wise CV

```text
A = age + spectral                       AUC = 0.729 ± 0.142
B = age + dwell_gradient                 AUC = 0.756 ± 0.132
C = age + spectral + dwell_gradient      AUC = 0.768 ± 0.128

C - A = +0.039 AUC
```

This is the strongest internal reason to keep going: the spatial dwell feature contributes a small positive mean increment beyond the three simple slowing features.

But it is **not external validation**. `dwell_gradient` was invented after inspecting ds004504. Cross-validation can reduce model-fitting leakage; it cannot erase the feature-selection history.

Full receipt: [`INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md`](INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md) and [`Results/phidwell_spectral_audit.json`](Results/phidwell_spectral_audit.json).

## 6. Age warning

The audit fixed the old parser issue (`Age` in the BIDS table versus lowercase `age` in legacy code). With age available:

```text
within AD: dwell_gradient vs age  rho = +0.1421, p = 0.4085
within CN: dwell_gradient vs age  rho = -0.5990, p = 0.000597
```

The strong age relationship among controls is a real warning. External evaluation must preserve the frozen age covariate and should report age balance / age-matched sensitivity. The current linear age term does not prove that all age-related structure has been removed.

## 7. The boring baseline: spectral slowing

Any Alzheimer's EEG feature must be tested against ordinary spectral slowing.

Freeze these baseline features:

```text
alpha_relative_power = power(8-13 Hz) / power(1-45 Hz)
theta_alpha_ratio    = power(4-8 Hz) / power(8-13 Hz)
peak_alpha_frequency = peak frequency in 7-13 Hz
```

References:

- Bruffaerts et al. (2025), *Diagnostic utility of electrophysiological markers for early and differential diagnosis of Alzheimer's, Frontotemporal, and Lewy Body dementias: A systematic review*. PMID 40379988.
- *EEG biomarkers in Alzheimer's and prodromal Alzheimer's: a comprehensive analysis of spectral and connectivity features*. PMID 39449097.
- Paitel et al. (2025), *Functional and effective EEG connectivity patterns in Alzheimer's disease and mild cognitive impairment: a systematic review*. PMID 40013094.

The comparator is a strength, not an embarrassment. If Φ-Dwell is useful it should show what it contributes **after** obvious spectral slowing is measured.

## 8. Raw versus cleaned EEG

Spatial phase measures can be sensitive to reference choice and ocular / muscle artifacts. The next internal robustness check is therefore:

```text
historical/raw-style pipeline
        versus
cleaned / derivative pipeline
```

Use [`phidwell_dwell_recompute.py`](phidwell_dwell_recompute.py) to recompute the unchanged dwell definition and then run the same spectral audit on the derivative EEG.

A result that only exists in the dirtier representation should not be promoted as a neural spatial-dynamics biomarker.

This robustness check still reuses the same people and cannot validate the Alzheimer claim.

## 9. PSI status

The Phase-Stability Index maps eigenmode power vectors onto an artificial radial canvas and measures Gerchberg–Saxton phase-recovery convergence.

Gerchberg–Saxton is a real optimization algorithm. The biological interpretation of its convergence count is currently unvalidated. PSI remains **quarantined exploratory work** until the simpler dwell feature independently replicates.

A historical PSI README also states that `p = 0.006` survives a Bonferroni threshold of approximately `0.003`; mathematically it does not (`0.006 > 0.003`).

## 10. Frozen external validation gate

### Primary question

> Does the existing dwell-gradient definition distinguish AD from controls in independent subjects, and does it add held-out information beyond ordinary spectral slowing?

### Frozen primary Φ-Dwell feature

```text
dwell_gradient only
```

### Models

```text
A: age + spectral baselines
B: age + dwell_gradient
C: age + spectral baselines + dwell_gradient
```

Use subject-wise held-out evaluation. The key incremental comparison is **C versus A**.

### No-rescue rule

After external labels are inspected, do not change:

- frequency bands;
- number of graph modes;
- graph sigma;
- word step;
- dwell definition;
- log transform;
- gradient direction;
- primary endpoint;
- spectral baseline definitions.

Other variants become new hypotheses for another held-out cohort.

### Verdicts

- `EXTERNAL_DWELL_GRADIENT_NULL` — no reliable external AD/CN signal.
- `REPLICATES_BUT_NO_INCREMENT_OVER_SPECTRAL_SLOWING` — representation is interesting, but not a new biomarker.
- `EXTERNAL_INCREMENTAL_SIGNAL` — frozen dwell gradient replicates and improves held-out performance beyond the spectral baseline.

Even the last verdict would establish a research signal, **not clinical diagnostic utility**.

## 11. Boundary

```text
INTERNAL AUDIT
- spectral slowing comparison: DONE
- within-AD MMSE: DONE (null)
- age parsing / age dependence: DONE (warning)
- raw vs cleaned: NEXT
- event/duration-matched PhysioNet task analysis: optional mechanism audit

EXTERNAL VALIDATION
- completely independent AD/control subjects
- frozen dwell_gradient
- frozen spectral baseline
- frozen age handling
- subject-wise held-out evaluation
```

That boundary is the main scientific upgrade to this repository.
