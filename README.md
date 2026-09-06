# Brain Metastability Analyzer Tool

> **2026 reality-audit reset:** this repository contains a real EEG state-space idea, but the historical Alzheimer's claims are exploratory and are **not** a validated diagnostic tool.

Live audit page: **https://anttiluode.github.io/BrainMetastabilityAnalyzerTool/**

## What Φ-Dwell actually measures

```text
EEG phase at sensors
    ↓
spatial graph-Laplacian basis built from electrode geometry
    ↓
dominant sensor-phase mode per frequency band
    ↓
dwell times / transitions / discrete multi-band states
```

The safest name is **spatial sensor-phase modes**. These are modes of the electrode-layout graph, not structural-connectome eigenmodes and not a holographic reconstruction of the brain.

The older repository used terms such as *holographic brain*, *criticality*, *grammar*, and *brain viscosity*. Those can be metaphors, but they are not evidence. See [`AUDIT_2026.md`](AUDIT_2026.md).

## What survives the audit

**Keep:**

- graph-Laplacian projection of multichannel EEG phase onto spatial sensor-layout modes;
- per-band dominant-mode dwell times and transition statistics;
- multi-band state words as an exploratory discretization;
- the already-discovered **dwell gradient** as a frozen candidate for independent replication.

**Repair / quarantine:**

- Pearson correlation of integer mode IDs is not a sound categorical coupling measure;
- `CV > 1` is dwell variability, not proof of criticality;
- Alzheimer bigram perplexity is trained and scored on the same sequence;
- PhysioNet “task doubles vocabulary” used unequal observation time and whole-run task labels despite alternating T0/T1/T2 epochs;
- PSI / Gerchberg–Saxton remains an unvalidated exploratory transform;
- the legacy age parser reads lowercase `age` even though ds004504 uses `Age`.

## Frozen dwell gradient

The later `brain_viscosity.py` branch contains a simple feature that does not need the viscosity story:

\[
g = \operatorname{slope}\left[\log(1+D_\delta),\log(1+D_\theta),\log(1+D_\alpha),\log(1+D_\beta),\log(1+D_\gamma)\right].
\]

Historical discovery on OpenNeuro `ds004504` reported strong-looking AD/CN separation and a pooled MMSE association. Those were discovery statistics from the same cohort on which the feature was developed.

## 2026 internal audit: raw-style EEG

The preregistered-style internal audit processed all **88 subjects with zero failures** and compared the frozen dwell gradient with ordinary spectral slowing.

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| dwell gradient | -0.4469 | -0.4164 | 0.000277 |
| alpha relative power | 0.0481 | 0.0755 | 0.004101 |
| theta / alpha ratio | 2.6013 | 1.8297 | 0.000993 |
| peak alpha frequency | 7.479 Hz | 8.664 Hz | 0.000124 |

Internal subject-wise CV:

```text
A = age + spectral                       AUC = 0.729 ± 0.142
B = age + dwell_gradient                 AUC = 0.756 ± 0.132
C = age + spectral + dwell_gradient      AUC = 0.768 ± 0.128

C - A = +0.039 AUC
```

That looked interesting, but it still reused the discovery cohort and therefore could not validate the feature.

Full raw-style receipt: [`INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md`](INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md).

## 2026 internal audit: cleaned / derivative EEG

The same frozen dwell transform was then recomputed on the dataset's derivative / cleaned EEG. Again, all **88 subjects** completed.

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| dwell gradient | -0.3551 | -0.3275 | 0.006175 |
| alpha relative power | 0.0494 | 0.0794 | 0.002351 |
| theta / alpha ratio | 2.5318 | 1.6045 | 0.0000732 |
| peak alpha frequency | 7.493 Hz | 8.681 Hz | 0.000155 |

So the univariate dwell difference **survives cleaning**, but its apparent incremental value does not:

```text
A = age + spectral                       AUC = 0.778 ± 0.121
B = age + dwell_gradient                 AUC = 0.694 ± 0.128
C = age + spectral + dwell_gradient      AUC = 0.777 ± 0.118

C - A = -0.001 AUC
```

Internal classification:

`INTERNAL_DERIVATIVE_DWELL_SEPARATION_NO_INCREMENT`

This is the main current result. Φ-Dwell still measures a disease-associated difference, but on cleaned EEG it does **not** improve held-out AD/CN discrimination beyond age + ordinary spectral slowing in this cohort.

Full cleaned-data receipt: [`INTERNAL_DERIVATIVE_AUDIT_RESULT_2026.md`](INTERNAL_DERIVATIVE_AUDIT_RESULT_2026.md).

## Severity claim: not supported

The pooled historical MMSE association does not survive as a within-disease severity result.

Raw-style:

```text
within AD:  rho = +0.215, p = 0.207
within FTD: rho = +0.121, p = 0.582
```

Cleaned:

```text
within AD:  rho = +0.269, p = 0.113
within FTD: rho = -0.009, p = 0.969
```

The defensible claim is group association in a discovery cohort, not cognitive-severity tracking.

## Age warning was preprocessing-sensitive

The raw-style audit found a strong dwell/age association in controls (`rho = -0.599`, `p = 0.000597`). After derivative preprocessing it vanished (`rho = -0.100`, `p = 0.607`). That makes the earlier age signal a robustness warning rather than a stable biological result.

## The boring competitor wins the current diagnostic contest

Alzheimer's EEG slowing is plainly visible in this dataset. In the cleaned analysis, age + alpha relative power + theta/alpha ratio + peak alpha frequency reached mean AUC `0.778`, while adding dwell gradient changed that to `0.777`.

So the current interpretation is:

> **Φ-Dwell may be an interesting spatial-dynamical representation of disease-related EEG change, but ds004504 does not show added diagnostic value beyond simple spectral slowing after cleaning.**

That is a useful result. It removes the strongest easy explanation for calling this a new cheap Alzheimer detector.

## Frozen external gate

The only decisive next test is on **completely independent AD/CN subjects** with the definition unchanged.

```text
Model A: age + spectral baselines
Model B: age + dwell_gradient
Model C: age + spectral baselines + dwell_gradient
```

Primary comparison: **C versus A on independent subjects**.

No changing bands, graph modes, graph sigma, word step, dwell definition, log transform, gradient direction, age handling, or primary endpoint after external labels are inspected.

External verdicts remain:

- `EXTERNAL_DWELL_GRADIENT_NULL`
- `REPLICATES_BUT_NO_INCREMENT_OVER_SPECTRAL_SLOWING`
- `EXTERNAL_INCREMENTAL_SIGNAL`

Even `EXTERNAL_INCREMENTAL_SIGNAL` would establish a research signal, not clinical diagnostic utility.

## Key files

- `eigenmode_metastability.py` — foundational dwell analysis.
- `phidwell_alzheimers.py` — historical discovery analyzer.
- `brain_viscosity.py` — origin of the dwell-gradient candidate.
- `phidwell_spectral_audit.py` — 2026 spectral-slowing audit.
- `phidwell_dwell_recompute.py` — frozen dwell recomputation on raw/derivative EEG.
- `AUDIT_2026.md` — methodological audit and frozen external gate.
- `INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md` — raw-style internal receipt.
- `INTERNAL_DERIVATIVE_AUDIT_RESULT_2026.md` — cleaned EEG robustness receipt.
- `Results/phidwell_spectral_audit.json` — raw-style machine-readable receipt.
- `Results/phidwell_spectral_audit_derivatives.json` — cleaned machine-readable receipt.
- `Alzheimers Phase Stability Index Test/` — quarantined exploratory PSI work.

## Clinical boundary

This repository is research software. It is **not a medical device, diagnostic test, or clinical decision tool**. No prospective diagnostic accuracy, acquisition-system generalization, or clinical utility has been established.

## License

MIT
