# Brain Viscosity Analyzer: β-Sieve × Φ-Dwell

This repository implements a novel analysis pipeline that extends the **Φ-Dwell** eigenmode metastability framework by introducing **viscosity-inspired gradients** across the frequency band hierarchy (δ → θ → α → β → γ). The tool identifies subtle disruptions in the brain's oscillatory cascade that are highly sensitive to neurodegeneration, particularly Alzheimer's disease.

## What It Does

The brain's frequency bands form a natural hierarchy: slower oscillations (delta, theta) modulate faster ones (alpha, beta, gamma). Healthy brains maintain a balanced, critical relationship across this hierarchy, with alpha often acting as the "pacemaker" for coherent spatial dynamics.

This tool computes **gradient metrics** (slopes across the five bands) from existing Φ-Dwell outputs:

- **dwell_gradient**: Slope of mean dwell time (log scale) from δ to γ.  
  Captures how stratified the temporal hierarchy is — shallow slope = flexible integration; steep slope = rigid separation between slow and fast rhythms.

- **self_rate_gradient**: Slope of self-transition rate (stickiness).  
  Measures how persistence changes across bands.

- **cv_gradient** and related measures: Slopes/differences in coefficient of variation (criticality regime).

- **viscosity_proxy**: Composite index combining hierarchy structure.

The script works in two modes:
- **Fast mode**: Loads a pre-computed `phidwell_alzheimer_results.json` and extracts gradients (seconds).
- **Full mode**: Processes raw EEG (`.set` files from OpenNeuro ds004504) to compute temporal roughness directly on eigenmode trajectories.

## Key Results on Alzheimer's/FTD Dataset (ds004504, N=88)

On resting eyes-closed EEG (19 channels, 2-minute recordings):

- **dwell_gradient** is the strongest biomarker found:
  - Kruskal-Wallis p = 0.0015 across CN/AD/FTD
  - AD vs CN pairwise p = 0.0003, FTD vs CN p = 0.036
  - Spearman correlation with MMSE: ρ = 0.408, p = 0.0001 (stronger than any original Φ-Dwell metric like vocabulary size, entropy, or Zipf α)

  Interpretation: Alzheimer's brains show a **steeper dwell gradient** (mean -0.447 vs -0.416 in controls). Slow bands preserve long dwells while mid/fast bands lose persistence — the hierarchy becomes more rigid and stratified.

- **Alpha band is the epicenter of pathology**:
  - Alpha CV (criticality): CN = 1.36 → AD = 1.14 (p ≈ 0.000)
  - Alpha self-rate (stickiness): CN = 0.876 → AD = 0.838 (p ≈ 0.001)
  - Alpha dwell time: CN = 5.37 → AD = 5.08 log-ms (p ≈ 0.001)

  This reveals classic "alpha slowing" as **loss of eigenmode stability** specifically in the dominant rhythm — alpha can no longer hold coherent spatial configurations.

- Beta and gamma also degrade (CV p ≈ 0.016–0.018), but **delta and theta are preserved** (p > 0.2). Disease selectively disrupts the middle of the cascade.

- **self_rate_gradient** trends similarly (AD vs CN p = 0.04, MMSE ρ = 0.249, p = 0.019).

- Other gradients (cv_gradient, hilo diffs) were non-significant — the full slope across all five bands captures more information than simple endpoint comparisons.

## Why This Matters

Traditional EEG markers in Alzheimer's focus on power slowing or coherence loss. Φ-Dwell originally showed increased vocabulary but reduced structure (flatter Zipf, subcritical shift). The viscosity gradients add a new dimension: **hierarchical rigidity** — the brain loses flexible integration across its natural frequency cascade, with alpha as the breaking point.

These metrics are complementary to spectral power and provide some of the strongest single-number separations on this dataset.

## Usage

```bash
# Fast analysis on existing Φ-Dwell results
python brain_viscosity.py phidwell_alzheimer_results.json

# Full pipeline on raw dataset (downloads if needed)
python brain_viscosity.py /path/to/ds004504/ --full
```

Outputs:
- Updated JSON with all gradient metrics per subject
- Console summary of group statistics and significance
- Optional figures (boxplots, profiles)

## References

- Original Φ-Dwell findings: Luode & Claude (2025) – Eigenmode Configuration Vocabulary Doubles During Motor Imagery
- Dataset: OpenNeuro ds004504 (36 AD, 23 FTD, 29 CN resting EEG)

This tool is exploratory research — results are promising but require replication on larger/high-density datasets. Contributions welcome!