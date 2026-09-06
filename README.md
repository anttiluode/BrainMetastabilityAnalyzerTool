# Brain Metastability Analyzer Tool

> **2026 reality-audit reset:** this repository contains a real EEG state-space idea, but the Alzheimer's claims are exploratory and are **not** yet a validated diagnostic tool.

This project decomposes scalp EEG phase patterns into spatial modes of the electrode geometry and measures how those mode configurations persist and change over time. The most defensible core is simple:

```text
EEG phase at sensors
    ↓
spatial graph-Laplacian basis
    ↓
dominant sensor-phase mode per frequency band
    ↓
dwell times / transitions / discrete multi-band states
```

The older repository text used terms such as *holographic brain*, *criticality*, *grammar*, and *brain viscosity*. Those names can be useful metaphors, but they are not evidence. The 2026 audit separates the measurable quantities from the story around them.

## What survives the audit

**Keep:**

- Graph-Laplacian projection of multichannel EEG phase onto spatial **sensor-layout modes**.
- Dominant-mode dwell times and transition statistics as descriptive EEG dynamics.
- Multi-band state words as an exploratory discrete representation.
- The previously discovered **dwell gradient** across delta → theta → alpha → beta → gamma as a candidate feature worth an independent replication test.

**Treat as exploratory / repair before reuse:**

- Vocabulary size, entropy, Zipf slope and top-word concentration: several are different summaries of the same empirical state-frequency distribution and are not independent biomarkers.
- Perplexity in `phidwell_alzheimers.py`: the current implementation builds and scores the bigram model on the same subject sequence, so it is descriptive training perplexity rather than held-out prediction.
- Cross-band coupling based on Pearson correlation of integer mode labels: mode IDs are categorical, so this metric is not invariant to arbitrary relabeling and should be replaced.
- `criticality_fraction`: the current rule is simply dwell-time CV > 1. CV is a useful variability statistic, but by itself it does not establish critical dynamics.
- PSI / Gerchberg–Saxton phase recovery: interesting mathematical probe, but currently an arbitrary transform of the eigenmode coefficients with no independent validation as a biological phase-stability measure.

See [`AUDIT_2026.md`](AUDIT_2026.md) for the detailed audit and frozen next test.

## The strongest candidate: dwell gradient

The later `brain_viscosity.py` branch derived a very simple feature from the existing per-band dwell measurements:

\[
g = \operatorname{slope}\left[\log(1+D_\delta),\log(1+D_\theta),\log(1+D_\alpha),\log(1+D_\beta),\log(1+D_\gamma)\right].
\]

On the discovery dataset (`OpenNeuro ds004504`, 36 AD, 23 FTD, 29 controls), the repository reported:

- Kruskal–Wallis across groups: `p = 0.0015`
- AD vs control: `p = 0.0003`
- pooled MMSE correlation: `rho = 0.408`, `p = 0.0001`

Those numbers are **discovery statistics**, not confirmation. The dwell-gradient feature was created after inspecting the same dataset, so the p-values cannot be interpreted as if the feature had been specified in advance.

The pooled MMSE correlation also needs care: in ds004504 all controls have MMSE = 30, so a feature that merely separates diagnosis groups can automatically correlate with MMSE. A proper severity question should be tested within AD (and/or within disease groups), not only across the pooled diagnostic sample.

## The boring competitor: ordinary spectral slowing

Any Alzheimer's EEG feature has to beat or add to the well-known spectral slowing signal. Dementia EEG commonly shows reduced dominant/alpha frequency and a shift toward relatively slower activity. That means a spatial-mode dwell effect can be interesting without being diagnostically new: it may simply be another view of the same alpha/theta slowing.

The next validation therefore must compare the frozen Φ-Dwell candidate against ordinary spectral baselines such as:

```text
alpha relative power
theta / alpha power ratio
peak alpha frequency
```

The important question is not merely *"does dwell gradient differ between AD and controls?"* It is:

> **Does frozen dwell gradient add held-out information beyond ordinary spectral slowing?**

If not, the representation may still be scientifically useful, but it should not be sold as a new cheap Alzheimer's biomarker.

## Important implementation findings

### 1. These are sensor-layout eigenmodes

The Laplacian is built from a Gaussian graph over electrode coordinates. It is not a structural-connectome eigenbasis and does not recover cortical anatomy. The safest wording is **spatial sensor-phase modes**.

### 2. Age parsing bug

The public dataset uses a column named `Age`; the original parser looks for lowercase `age`. This is why existing saved results contain `age: null`. No age-adjusted claim should be made from those saved results until the parser is repaired and the analysis rerun.

### 3. Current cross-band coupling is mathematically unsound

The code correlates dominant mode IDs as numbers. Renaming mode 1 ↔ 6 changes Pearson correlation even though the categorical brain-state sequence is unchanged. Replace this with a label-invariant statistic such as mutual information, normalized mutual information, or a measure on the continuous eigenmode coefficient vectors.

### 4. The task-vocabulary result is confounded

The current PhysioNet grammar scripts concatenate roughly two short baseline runs for REST but six longer motor-imagery runs for TASK, and they label whole motor-imagery recordings as TASK even though those recordings alternate T0 rest and T1/T2 task epochs. Vocabulary size increases with observation time, so the published "task doubles vocabulary" result requires an equal-duration, event-conditioned rerun.

### 5. Raw versus cleaned EEG must be checked

Spatial phase measures are sensitive to reference choice, ocular/muscle contamination and other sensor-space structure. The Alzheimer's result should be rerun both on the raw input used historically and on the dataset's cleaned/preprocessed derivative where available.

## Frozen next gate

Before looking at a new Alzheimer's EEG cohort, freeze this primary question:

> **Does the existing dwell-gradient definition distinguish AD from controls in independent subjects, without changing its definition, and does it add information beyond ordinary spectral slowing?**

Recommended primary feature:

```text
dwell_gradient only
```

Recommended comparison:

```text
Model A: spectral baselines only
Model B: dwell_gradient only
Model C: spectral baselines + dwell_gradient
```

Use subject-wise held-out evaluation. No tuning of bands, number of modes, word step, dwell definition, or gradient direction after seeing the external labels.

Interpretation:

- **Fails externally:** retire the Alzheimer's biomarker claim.
- **Replicates but adds nothing beyond spectral slowing:** keep as an alternative representation, not a new biomarker.
- **Replicates and improves held-out performance beyond spectral baselines:** strong reason to continue.

## Current files

- `eigenmode_metastability.py` — foundational dwell analysis.
- `phidwell_deep_analyzer.py` — exploratory configuration-space analysis.
- `phidwell_grammar_decoder.py` — state vocabulary analysis.
- `phidwell_perplexity.py` — rest/task n-gram analysis; needs equal-duration event conditioning for a clean task test.
- `phidwell_alzheimers.py` — discovery analysis on OpenNeuro ds004504; requires audit fixes before new claims.
- `brain_viscosity.py` — contains the dwell-gradient candidate; the "viscosity" analogy is not required for the metric.
- `Alzheimers Phase Stability Index Test/` — PSI experiment; quarantined as exploratory until simpler metrics replicate.
- `Results/` — historical outputs. Treat them as discovery receipts, not independent validation.

## Clinical boundary

This repository is research software. It is **not a medical device, diagnostic test, or clinical decision tool**. The existing results come from public research datasets and have not established prospective diagnostic accuracy, generalization across acquisition systems, or added value beyond standard EEG markers.

## Live page

GitHub Pages is enabled for this repository. The static audit page is at:

**https://anttiluode.github.io/BrainMetastabilityAnalyzerTool/**

## License

MIT
