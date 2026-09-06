# Φ-Dwell cleaned/derivative robustness result — 2026

Status: **internal robustness result, not external validation**.

This receipt uses the same 88 OpenNeuro `ds004504` participants as the historical discovery work, but recomputes the frozen Φ-Dwell dwell feature from the dataset's derivative / cleaned EEG and reruns the already-frozen spectral comparison.

## Result in one sentence

> The AD-versus-control dwell-gradient difference survives cleaned EEG as a univariate group effect, but its apparent incremental value beyond ordinary spectral slowing disappears.

Internal classification:

`INTERNAL_DERIVATIVE_DWELL_SEPARATION_NO_INCREMENT`

This is deliberately **not** one of the external-validation verdicts.

## Data receipt

All 88 subjects were processed successfully.

| Feature | AD mean | CN mean | two-sided p |
|---|---:|---:|---:|
| `dwell_gradient` | -0.3551 | -0.3275 | 0.006175 |
| alpha relative power | 0.0494 | 0.0794 | 0.002351 |
| theta / alpha ratio | 2.5318 | 1.6045 | 0.0000732 |
| peak alpha frequency | 7.493 Hz | 8.681 Hz | 0.000155 |

The dwell effect therefore does not vanish after derivative preprocessing. However, the conventional slowing features are at least as strong and generally stronger in this representation.

## Incremental prediction result

Repeated subject-wise internal CV:

```text
A = age + spectral                       AUC = 0.778 ± 0.121
B = age + dwell_gradient                 AUC = 0.694 ± 0.128
C = age + spectral + dwell_gradient      AUC = 0.777 ± 0.118

C - A = -0.00125 AUC
```

The cleaned-data result therefore removes the earlier internal `+0.039` AUC increment seen in the raw-style analysis.

The most defensible interpretation is:

- the spatial dwell measurement is not pure sensor dirt, because the AD/CN group difference survives derivative preprocessing;
- but the cleaned dwell feature does **not** add useful discrimination beyond age + ordinary spectral slowing in this cohort;
- the raw-style incremental gain was preprocessing-sensitive and should not be treated as evidence for a novel Alzheimer biomarker.

## Severity result remains null

Within-disease MMSE associations remain non-significant:

```text
within AD:  rho = +0.269, p = 0.113, n = 36
within FTD: rho = -0.009, p = 0.969, n = 23
```

So the historical pooled MMSE association still does not support a clean disease-severity claim.

## Age warning changed after cleaning

The raw-style audit found a strong control-group dwell/age association. It does **not** survive derivative preprocessing:

```text
within AD: dwell vs age  rho = +0.276, p = 0.103
within CN: dwell vs age  rho = -0.100, p = 0.607
```

That makes the earlier control-age effect preprocessing-sensitive rather than a stable property of the current measure.

## Raw-style versus derivative summary

| Quantity | Raw-style | Derivative / cleaned |
|---|---:|---:|
| dwell AD-CN p | 0.000277 | 0.006175 |
| spectral Model A AUC | 0.729 | 0.778 |
| dwell Model B AUC | 0.756 | 0.694 |
| combined Model C AUC | 0.768 | 0.777 |
| C - A | +0.039 | -0.001 |
| within-AD MMSE p | 0.207 | 0.113 |

The key scientific change is not the univariate p-value. It is the loss of incremental held-out value after cleaning.

## What this does and does not establish

This supports continued interest in Φ-Dwell as an EEG spatial-state representation. It does **not** support presenting Φ-Dwell as a new cheap Alzheimer detector on the basis of ds004504.

The next decisive question remains external:

> Does the unchanged dwell-gradient definition reproduce in completely independent AD/CN subjects, and does it add held-out information beyond the same frozen spectral baseline?

No parameter retuning on ds004504 should be used to rescue the lost incremental effect.

## Receipts

- `Results/phidwell_dwell_derivatives.json`
- `Results/phidwell_dwell_derivatives_meta.json`
- `Results/phidwell_spectral_audit_derivatives.json`
- `Results/phidwell_spectral_audit_derivatives.csv`
