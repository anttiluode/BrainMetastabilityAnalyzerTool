# Φ-Dwell internal spectral-slowing audit — 2026 result

Status: **internal discovery-cohort audit, not external validation**.

Source receipt: [`Results/phidwell_spectral_audit.json`](Results/phidwell_spectral_audit.json)

Dataset: OpenNeuro `ds004504` (36 AD, 23 FTD, 29 cognitively normal controls). The audit processed all 88 subjects with zero failures using 120 s of raw-style EEG per subject.

## Frozen question

The audit was written before this output was inspected to ask whether the already-discovered Φ-Dwell `dwell_gradient` contains information beyond ordinary Alzheimer-related spectral slowing.

```text
Φ-Dwell:
  dwell_gradient = slope(log(1 + band_mean_dwell))
                   across delta → theta → alpha → beta → gamma

Spectral baselines:
  alpha_relative_power = P(8-13 Hz) / P(1-45 Hz)
  theta_alpha_ratio    = P(4-8 Hz) / P(8-13 Hz)
  peak_alpha_frequency = PSD peak in 7-13 Hz
```

The predictive models were:

```text
A = age + spectral baselines
B = age + dwell_gradient
C = age + spectral baselines + dwell_gradient
```

## Group results: AD versus controls

| Feature | AD mean | CN mean | two-sided Mann–Whitney p |
|---|---:|---:|---:|
| `dwell_gradient` | -0.4469 | -0.4164 | 0.000277 |
| alpha relative power | 0.0481 | 0.0755 | 0.004101 |
| theta / alpha ratio | 2.6013 | 1.8297 | 0.000993 |
| peak alpha frequency | 7.479 Hz | 8.664 Hz | 0.000124 |

The ordinary spectral-slowing baselines are therefore plainly present in this cohort. Peak alpha frequency is at least as striking a group marker as the Φ-Dwell feature by this simple univariate comparison.

The dwell-gradient group effect nevertheless remains visible in the audit implementation, which is useful as a reproducibility receipt for the historical discovery.

## Severity sanity check

The old repository highlighted a pooled MMSE association (`rho ≈ 0.408`). The audit asked the more appropriate within-diagnosis question.

```text
within AD:  dwell_gradient vs MMSE
rho = 0.2154, p = 0.2072, n = 36

within FTD: dwell_gradient vs MMSE
rho = 0.1210, p = 0.5824, n = 23
```

So the current data do **not** support the claim that dwell gradient tracks cognitive severity within AD. The historical pooled MMSE association was substantially entangled with diagnostic-group separation.

## Subject-wise repeated cross-validation

The audit used 5-fold stratified CV repeated 20 times (100 held-out folds), with scaling and logistic regression fit inside each training fold.

```text
A: age + spectral                       mean AUC = 0.729 ± 0.142
B: age + dwell_gradient                 mean AUC = 0.756 ± 0.132
C: age + spectral + dwell_gradient      mean AUC = 0.768 ± 0.128

C - A = +0.039 AUC
```

This is the most interesting internal result: adding the spatial dwell feature improved mean held-out AUC by about **0.039** over age plus the three simple spectral features.

But this is **not** an independent confirmation. `dwell_gradient` was invented after inspecting this same ds004504 cohort. Cross-validation can reduce model-fitting leakage; it cannot undo feature-selection history. The +0.039 increment is therefore a reason to perform the frozen external test, not a validated effect size.

## Age warning exposed by the audit

With age now parsed correctly:

```text
within AD: dwell_gradient vs age
rho = +0.1421, p = 0.4085

within CN: dwell_gradient vs age
rho = -0.5990, p = 0.000597
```

The strong age association among controls is a real warning. It means external evaluation must preserve the frozen age covariate and should report age balance / age-matched sensitivity. A linear age covariate in the current model does not prove that all age-related structure has been removed.

## Current interpretation

The internal audit does **not** reduce Φ-Dwell to ordinary spectral slowing, but neither does it establish a new Alzheimer biomarker.

What survived:

> In the original discovery cohort, the frozen dwell-gradient remains different between AD and controls and adds a small positive internal-CV increment beyond three simple spectral baselines.

What did not survive:

> The pooled MMSE result does not become a within-AD severity relationship.

What remains decisive:

> A completely independent cohort, using the already-frozen dwell-gradient and spectral definitions, must determine whether the incremental signal generalizes.

## Next internal robustness check

Before external validation, a useful non-confirmatory robustness test is to recompute the same dwell feature on the dataset's cleaned / derivative EEG and compare it with the raw-style receipt. This is a preprocessing robustness check only; it still reuses the same people.

No bands, graph settings, dwell definition, gradient direction, or primary external endpoint should be changed based on this result.
