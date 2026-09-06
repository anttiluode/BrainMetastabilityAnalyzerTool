# Brain Metastability Analyzer Tool

> **2026 reality-audit reset:** this repository contains a real EEG state-space idea, but the Alzheimer's claims remain exploratory and are **not** a validated diagnostic tool.

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
- the already-discovered **dwell gradient** as one candidate worth a frozen independent replication.

**Repair / quarantine:**

- Pearson correlation of integer mode IDs is not a sound categorical coupling measure;
- `CV > 1` is dwell variability, not proof of criticality;
- Alzheimer bigram perplexity is trained and scored on the same sequence;
- PhysioNet “task doubles vocabulary” used unequal observation time and whole-run task labels despite alternating T0/T1/T2 epochs;
- PSI / Gerchberg–Saxton remains an unvalidated exploratory transform;
- the legacy age parser reads lowercase `age` even though ds004504 uses `Age`.

## Strongest candidate: dwell gradient

The later `brain_viscosity.py` branch contains a simple feature that does not need the viscosity story:

\[
g = \operatorname{slope}\left[\log(1+D_\delta),\log(1+D_\theta),\log(1+D_\alpha),\log(1+D_\beta),\log(1+D_\gamma)\right].
\]

Historical discovery on OpenNeuro `ds004504` reported AD/CN separation around `p ≈ 0.0003` and a pooled MMSE association around `rho ≈ 0.408`. Those were discovery statistics from the same cohort on which the feature was developed.

## 2026 internal spectral-slowing audit

The audit was written before its output was inspected. It processed all **88 subjects with zero failures** and compared the frozen dwell gradient with ordinary spectral slowing.

### AD versus controls

| Feature | AD mean | CN mean | p |
|---|---:|---:|---:|
| dwell gradient | -0.4469 | -0.4164 | 0.000277 |
| alpha relative power | 0.0481 | 0.0755 | 0.004101 |
| theta / alpha ratio | 2.6013 | 1.8297 | 0.000993 |
| peak alpha frequency | 7.479 Hz | 8.664 Hz | 0.000124 |

So ordinary spectral slowing is plainly present. Peak alpha frequency is at least as striking a simple group marker as dwell gradient in this cohort.

### Severity check

The pooled MMSE result did **not** become a within-disease severity result:

```text
within AD:  dwell_gradient vs MMSE  rho = 0.215, p = 0.207
within FTD: dwell_gradient vs MMSE  rho = 0.121, p = 0.582
```

That strongly suggests the historical pooled MMSE correlation was substantially driven by diagnostic-group separation.

### Internal subject-wise cross-validation

```text
A = age + spectral                       AUC = 0.729 ± 0.142
B = age + dwell_gradient                 AUC = 0.756 ± 0.132
C = age + spectral + dwell_gradient      AUC = 0.768 ± 0.128

C - A = +0.039 AUC
```

This is interesting, but it is **not external validation**. Dwell gradient was invented after looking at the same ds004504 cohort, so cross-validation cannot erase feature-selection history. The +0.039 increment is a reason to perform the frozen external test, not a validated effect size.

Full receipt: [`INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md`](INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md) and [`Results/phidwell_spectral_audit.json`](Results/phidwell_spectral_audit.json).

## Age warning

With age parsed correctly, dwell gradient showed:

```text
within AD: rho = +0.142, p = 0.409
within CN: rho = -0.599, p = 0.000597
```

The strong control-group age association is a real warning. External evaluation must preserve age handling and should report age balance / age-matched sensitivity.

## The boring competitor: spectral slowing

Any Alzheimer's EEG feature has to add something beyond well-known slowing of EEG frequency content. The frozen baseline is:

```text
alpha_relative_power = P(8-13 Hz) / P(1-45 Hz)
theta_alpha_ratio    = P(4-8 Hz) / P(8-13 Hz)
peak_alpha_frequency = PSD peak in 7-13 Hz
```

The important question is now:

> **Does frozen dwell gradient add held-out information beyond ordinary spectral slowing in completely independent subjects?**

## Frozen external gate

```text
Model A: age + spectral baselines
Model B: age + dwell_gradient
Model C: age + spectral baselines + dwell_gradient
```

Primary comparison: **C versus A on independent subjects**.

No changing bands, graph modes, word step, dwell definition, log transform, gradient direction, or primary endpoint after external labels are inspected.

Verdicts:

- `EXTERNAL_DWELL_GRADIENT_NULL`
- `REPLICATES_BUT_NO_INCREMENT_OVER_SPECTRAL_SLOWING`
- `EXTERNAL_INCREMENTAL_SIGNAL`

Even the last verdict would establish a research signal, not clinical diagnostic utility.

## Next internal robustness test: cleaned EEG

A useful non-confirmatory check is to recompute the **same** dwell feature on the dataset's derivative / cleaned EEG.

```bat
python3.13 phidwell_dwell_recompute.py "E:\PATH\TO\ds004504" ^
  --use-derivatives ^
  --out "Results\phidwell_dwell_derivatives.json"

python3.13 phidwell_spectral_audit.py "E:\PATH\TO\ds004504" ^
  --results "Results\phidwell_dwell_derivatives.json" ^
  --use-derivatives ^
  --out "Results\phidwell_spectral_audit_derivatives.json"
```

This still uses the same people, so it is a preprocessing robustness receipt only.

## Key files

- `eigenmode_metastability.py` — foundational dwell analysis.
- `phidwell_alzheimers.py` — historical discovery analyzer.
- `brain_viscosity.py` — origin of the dwell-gradient candidate.
- `phidwell_spectral_audit.py` — 2026 spectral-slowing internal audit.
- `phidwell_dwell_recompute.py` — exact frozen dwell recomputation for raw/derivative robustness.
- `AUDIT_2026.md` — methodological audit and frozen external gate.
- `INTERNAL_SPECTRAL_AUDIT_RESULT_2026.md` — current internal audit receipt.
- `Alzheimers Phase Stability Index Test/` — quarantined exploratory PSI work.

## Clinical boundary

This repository is research software. It is **not a medical device, diagnostic test, or clinical decision tool**. No prospective diagnostic accuracy, acquisition-system generalization, or clinical utility has been established.

## License

MIT
