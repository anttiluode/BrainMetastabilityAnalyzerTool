# BrainMetastabilityAnalyzerTool — 2026 Reality Audit

Status: **discovery-stage research code**. The repository contains a defensible EEG state-space representation, but its historical Alzheimer/FTD language overstates what the existing experiments establish.

This document freezes the distinction between **measurement**, **discovery**, and **validation** before more data are inspected.

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
- `dwell_gradient` as a single candidate feature for independent replication.

### EXPLORATORY

- vocabulary size;
- entropy;
- Zipf exponent;
- top-5 concentration;
- training perplexity;
- PSI / Gerchberg–Saxton convergence;
- the `viscosity` interpretation.

Several vocabulary metrics summarize the same state-count distribution and should not be presented as independent biomarkers.

### REPAIR / RETIRE

**Categorical Pearson coupling.** The current Alzheimer's analyzer computes Pearson correlation between dominant mode IDs. Mode IDs are labels. Relabeling mode `1` as `6` changes Pearson correlation without changing the categorical sequence. The metric is therefore not label-invariant. Replace it with mutual information / normalized mutual information or operate on the continuous eigenmode coefficient vectors.

**Criticality from CV > 1.** Dwell-time coefficient of variation is a legitimate statistic. A threshold of `CV > 1` does not by itself demonstrate a critical phase transition. Use the descriptive name `dwell_cv` unless stronger criticality tests are added.

**In-sample bigram perplexity.** `phidwell_alzheimers.py` builds the subject bigram counts and scores the same sequence. This is descriptive training perplexity, not predictive generalization. Use blocked held-out scoring for predictive language.

## 3. The task-vocabulary result is confounded

The historical PhysioNet task analysis compares unequal observation amounts:

- REST concatenates the two baseline recordings;
- TASK concatenates six motor-imagery recordings;
- the task recordings are much longer than the baseline recordings;
- the whole motor-imagery file is labeled TASK even though the EDF annotations alternate `T0` rest and `T1/T2` task blocks.

Vocabulary size naturally grows with sample duration. Therefore the statement that "task doubles the vocabulary" is not currently established.

A clean rerun must use **equal-duration event-conditioned T0 versus T1/T2 samples within the same recordings**, with the state representation fixed before outcome inspection.

## 4. Alzheimer's discovery dataset

Historical Alzheimer's work used OpenNeuro `ds004504`:

- 36 Alzheimer's disease (AD)
- 23 frontotemporal dementia (FTD)
- 29 cognitively normal controls (CN)
- 19 scalp EEG channels
- resting eyes closed

The original analyzer tested many features on this one cohort. Reported p-values should therefore be treated as **discovery statistics**, not external confirmation.

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

This is strong enough to justify an external test, but not strong enough to call the feature a validated biomarker.

## 5. MMSE confound

The public participant table assigns **MMSE = 30 to every control subject**. AD and FTD have lower and variable MMSE scores.

Therefore a pooled correlation between an EEG feature and MMSE can arise simply because the EEG feature separates diagnostic groups.

The audit requires:

- pooled MMSE correlation: descriptive only;
- **within-AD MMSE correlation**: severity test;
- optional within-FTD correlation: secondary;
- age correctly parsed and reported.

The legacy parser looks for lowercase `age`, while the dataset column is `Age`. Existing saved JSON therefore contains `age: null` and should not be described as age-adjusted.

## 6. The boring baseline: spectral slowing

Any Alzheimer's EEG feature must be tested against ordinary spectral slowing.

The literature consistently reports dementia-related slowing of EEG frequency content. A 2025 systematic review of 70 biomarker-proven dementia studies described spectral slowing as a common finding across dementias and highlighted EEG's potential as a low-cost, accessible modality. Other recent work reports higher theta-related power and theta/alpha or theta/beta ratios in AD, while alpha-band connectivity effects are among the most recurrent network findings.

References:

- Bruffaerts et al. (2025), *Diagnostic utility of electrophysiological markers for early and differential diagnosis of Alzheimer's, Frontotemporal, and Lewy Body dementias: A systematic review*. PMID 40379988. https://pubmed.ncbi.nlm.nih.gov/40379988/
- *EEG biomarkers in Alzheimer's and prodromal Alzheimer's: a comprehensive analysis of spectral and connectivity features*. PMID 39449097. https://pubmed.ncbi.nlm.nih.gov/39449097/
- Paitel et al. (2025), *Functional and effective EEG connectivity patterns in Alzheimer's disease and mild cognitive impairment: a systematic review*. PMID 40013094. https://pubmed.ncbi.nlm.nih.gov/40013094/

The boring comparator is a strength, not an embarrassment. If Φ-Dwell is useful it should show what it contributes **after** the obvious spectral signal is measured.

Freeze these baseline features for the audit:

```text
alpha_relative_power = power(8-13 Hz) / power(1-45 Hz)
theta_alpha_ratio    = power(4-8 Hz) / power(8-13 Hz)
peak_alpha_frequency = peak frequency in 7-13 Hz
```

## 7. Raw versus cleaned EEG

Spatial phase measures can be sensitive to reference choice and ocular / muscle artifacts. The ds004504 ecosystem includes analyses using A1/A2 rereferencing, 0.5–45 Hz filtering, ASR, ICA and artifact-component rejection.

A useful internal robustness receipt is therefore:

```text
historical/raw-style pipeline
        versus
cleaned / derivative pipeline
```

A result that only exists in the dirtier representation should not be promoted as a neural spatial-dynamics biomarker.

## 8. PSI status

The Phase-Stability Index maps eigenmode power vectors onto an artificial radial canvas and measures Gerchberg–Saxton phase-recovery convergence.

Gerchberg–Saxton is a real optimization algorithm. The biological interpretation of its convergence count is currently unvalidated. PSI remains **quarantined exploratory work** until the much simpler dwell feature independently replicates.

A historical PSI README also states that `p = 0.006` survives a Bonferroni threshold of approximately `0.003`; mathematically it does not (`0.006 > 0.003`). That statement should not be used as evidence.

## 9. Frozen validation gate

### Primary question

> Does the **existing** dwell-gradient definition distinguish AD from controls in independent subjects, and does it add held-out information beyond ordinary spectral slowing?

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

Other variants may be explored only as explicitly new hypotheses on another held-out cohort.

### Verdicts

- `EXTERNAL_DWELL_GRADIENT_NULL` — no reliable external AD/CN signal.
- `REPLICATES_BUT_NO_INCREMENT_OVER_SPECTRAL_SLOWING` — spatial representation is interesting, but not a new biomarker.
- `EXTERNAL_INCREMENTAL_SIGNAL` — frozen dwell gradient replicates and improves held-out performance beyond the spectral baseline.

Even the last verdict would establish a research signal, **not** clinical diagnostic utility.

## 10. Internal audit versus external validation

Re-running ds004504 with better statistics is worthwhile for debugging and understanding the mechanism. It cannot turn the discovery cohort into an independent replication cohort.

The next two levels are deliberately separate:

```text
INTERNAL AUDIT
- fix age
- compare spectral baselines
- within-AD MMSE
- raw vs cleaned
- event/duration-match PhysioNet task analysis

EXTERNAL VALIDATION
- completely independent AD/control subjects
- frozen dwell_gradient
- frozen spectral baseline
- subject-wise held-out evaluation
```

That boundary is the main scientific upgrade to this repository.
