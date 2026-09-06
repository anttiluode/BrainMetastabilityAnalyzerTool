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
- `dwell_gradient` as a single frozen candidate for independent replication.

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

## 5. Internal raw-style spectral audit — completed

The frozen audit processed all 88 subjects with zero failures and compared dwell gradient with three ordinary spectral baselines.

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| `dwell_gradient` | -0.4469 | -0.4164 | 0.000277 |
| alpha relative power | 0.0481 | 0.0755 | 0.004101 |
| theta / alpha ratio | 2.6013 | 1.8297 | 0.000993 |
| peak alpha frequency | 7.479 Hz | 8.664 Hz | 0.000124 |

Internal repeated subject-wise CV:

```text
A = age + spectral                       AUC = 0.729 ± 0.142
B = age + dwell_gradient                 AUC = 0.756 ± 0.132
C = age + spectral + dwell_gradient      AUC = 0.768 ± 0.128

C - A = +0.039 AUC
```

This was interesting but not external validation. The feature was developed on the same cohort.

Raw-style receipt: [`INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md`](INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md).

## 6. Internal derivative / cleaned EEG audit — completed

The same frozen spatial-phase and dwell transform was recomputed on the dataset's derivative / cleaned EEG, again with 88/88 subjects processed.

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| `dwell_gradient` | -0.3551 | -0.3275 | 0.006175 |
| alpha relative power | 0.0494 | 0.0794 | 0.002351 |
| theta / alpha ratio | 2.5318 | 1.6045 | 0.0000732 |
| peak alpha frequency | 7.493 Hz | 8.681 Hz | 0.000155 |

The univariate dwell difference therefore survives preprocessing, although it weakens.

The decisive internal comparison changes much more strongly:

```text
A = age + spectral                       AUC = 0.778 ± 0.121
B = age + dwell_gradient                 AUC = 0.694 ± 0.128
C = age + spectral + dwell_gradient      AUC = 0.777 ± 0.118

C - A = -0.00125 AUC
```

Internal classification:

`INTERNAL_DERIVATIVE_DWELL_SEPARATION_NO_INCREMENT`

This means the disease-associated dwell difference is not simply destroyed by cleaning, but its apparent **incremental diagnostic value beyond ordinary slowing is preprocessing-sensitive and disappears in the cleaned representation**.

That substantially weakens the case for Φ-Dwell as a new cheap Alzheimer biomarker on ds004504. The representation may still be scientifically useful as a spatial-dynamical view of disease-related EEG change.

Cleaned receipt: [`INTERNAL_DERIVATIVE_AUDIT_RESULT_2026.md`](INTERNAL_DERIVATIVE_AUDIT_RESULT_2026.md).

## 7. Severity claim — not supported

The historical pooled MMSE association does not become a reliable within-disease severity effect.

Raw-style:

```text
within AD:  rho = +0.215, p = 0.207
within FTD: rho = +0.121, p = 0.582
```

Derivative / cleaned:

```text
within AD:  rho = +0.269, p = 0.113
within FTD: rho = -0.009, p = 0.969
```

The old pooled MMSE relationship was entangled with diagnostic-group separation. The repository should not claim that dwell gradient tracks cognitive severity.

## 8. Age warning became preprocessing-sensitive

The raw-style audit found:

```text
within AD: dwell vs age  rho = +0.142, p = 0.409
within CN: dwell vs age  rho = -0.599, p = 0.000597
```

After derivative preprocessing:

```text
within AD: dwell vs age  rho = +0.276, p = 0.103
within CN: dwell vs age  rho = -0.100, p = 0.607
```

The strong control-age relationship therefore does not survive preprocessing and should not be treated as a stable biological finding.

Age still remains a frozen covariate for external evaluation.

## 9. The boring baseline: spectral slowing

Any Alzheimer's EEG feature must be tested against ordinary spectral slowing.

Frozen baselines:

```text
alpha_relative_power = power(8-13 Hz) / power(1-45 Hz)
theta_alpha_ratio    = power(4-8 Hz) / power(8-13 Hz)
peak_alpha_frequency = peak frequency in 7-13 Hz
```

The cleaned internal result currently favors the boring explanation for diagnostic performance: spectral features alone reach mean AUC `0.778`, and adding dwell changes that to `0.777`.

This does **not** prove dwell and spectral slowing are mathematically identical. It shows that the current dwell feature contributes no additional held-out discrimination in this cleaned cohort under the frozen model.

References:

- Bruffaerts et al. (2025), *Diagnostic utility of electrophysiological markers for early and differential diagnosis of Alzheimer's, Frontotemporal, and Lewy Body dementias: A systematic review*. PMID 40379988.
- *EEG biomarkers in Alzheimer's and prodromal Alzheimer's: a comprehensive analysis of spectral and connectivity features*. PMID 39449097.
- Paitel et al. (2025), *Functional and effective EEG connectivity patterns in Alzheimer's disease and mild cognitive impairment: a systematic review*. PMID 40013094.

## 10. PSI status

The Phase-Stability Index maps eigenmode power vectors onto an artificial radial canvas and measures Gerchberg–Saxton phase-recovery convergence.

Gerchberg–Saxton is a real optimization algorithm. The biological interpretation of its convergence count is currently unvalidated. PSI remains quarantined exploratory work.

A historical PSI README also states that `p = 0.006` survives a Bonferroni threshold of approximately `0.003`; mathematically it does not (`0.006 > 0.003`).

## 11. Frozen external validation gate

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
- spectral baseline definitions;
- age handling.

Other variants become new hypotheses for another held-out cohort.

### External verdicts

- `EXTERNAL_DWELL_GRADIENT_NULL` — no reliable external AD/CN signal.
- `REPLICATES_BUT_NO_INCREMENT_OVER_SPECTRAL_SLOWING` — representation is interesting, but not a new biomarker.
- `EXTERNAL_INCREMENTAL_SIGNAL` — frozen dwell gradient replicates and improves held-out performance beyond the spectral baseline.

Even the last verdict would establish a research signal, **not clinical diagnostic utility**.

## 12. Current boundary

```text
INTERNAL AUDIT
- spectral slowing comparison: DONE
- within-AD MMSE: DONE (null)
- age parsing: DONE
- raw vs cleaned: DONE
- cleaned incremental value: NONE under current frozen model
- event/duration-matched PhysioNet task analysis: optional mechanism audit

EXTERNAL VALIDATION
- completely independent AD/control subjects
- frozen dwell_gradient
- frozen spectral baseline
- frozen age handling
- subject-wise held-out evaluation
```

The next Alzheimer test should use new people. Do not retune ds004504 to recover the lost `+0.039` increment.
