#!/usr/bin/env python3
"""
2026 internal reality-audit for the Φ-Dwell Alzheimer's discovery cohort.

This script does NOT validate the biomarker. It re-examines the already-used
OpenNeuro ds004504 cohort with a frozen, boring comparator: spectral slowing.

It combines the historical per-subject Φ-Dwell result JSON with three ordinary
spectral EEG features computed from the same subject recordings:

    alpha_relative_power = power(8-13 Hz) / power(1-45 Hz)
    theta_alpha_ratio    = power(4-8 Hz) / power(8-13 Hz)
    peak_alpha_frequency = peak of mean PSD in 7-13 Hz

The Φ-Dwell candidate is fixed to the already-discovered dwell gradient:

    slope(log(1 + mean_dwell_band))
    for delta -> theta -> alpha -> beta -> gamma

Primary INTERNAL question:
    Does dwell_gradient add cross-validated AD-vs-CN discrimination beyond
    age + ordinary spectral slowing on the discovery dataset?

Because dwell_gradient was discovered on this same dataset, the answer is an
internal sanity check only. It is not independent replication.

Usage:
    python phidwell_spectral_audit.py "path/to/ds004504" \
        --results "path/to/phidwell_alzheimer_results.json"

For a cleaned/derivative rerun, first generate a matching Φ-Dwell result JSON
from that representation, then run:

    python phidwell_spectral_audit.py "path/to/ds004504" \
        --results "path/to/derivative_phidwell_results.json" \
        --use-derivatives

Outputs JSON + CSV next to --out.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import signal, stats

try:
    import mne
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mne is required: pip install mne") from exc

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc


BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
SEED = 20260906


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_participants(dataset_path: str) -> Dict[str, dict]:
    """Read ds004504 participants.tsv, accepting both Age and age."""
    path = Path(dataset_path) / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"participants.tsv not found: {path}")

    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sid = str(row.get("participant_id", "")).strip()
            if not sid:
                continue
            group_raw = str(row.get("Group", row.get("group", ""))).strip()
            group = {"A": "AD", "C": "CN", "F": "FTD"}.get(group_raw, group_raw)
            out[sid] = {
                "group": group,
                "age": _as_float(row.get("Age", row.get("age"))),
                "mmse": _as_float(row.get("MMSE", row.get("mmse"))),
            }
    return out


def dwell_gradient(subject_result: dict) -> float:
    dwell = subject_result.get("band_mean_dwell", {})
    values = []
    for band in BAND_ORDER:
        if band not in dwell:
            raise KeyError(f"missing band_mean_dwell[{band!r}]")
        values.append(np.log(float(dwell[band]) + 1.0))
    slope, _, _, _, _ = stats.linregress(np.arange(5, dtype=float), values)
    return float(slope)


def find_subject_file(dataset_path: str, subject_id: str, use_derivatives: bool) -> Optional[str]:
    base = Path(dataset_path) / "derivatives" if use_derivatives else Path(dataset_path)
    subject_root = base / subject_id

    patterns = ["**/*.set", "**/*.edf", "**/*.fif"]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(str(subject_root / pattern), recursive=True))

    if not candidates and use_derivatives:
        # Some derivative packages insert pipeline folders between derivatives/ and sub-*.
        for ext in ("set", "edf", "fif"):
            candidates.extend(
                glob.glob(str(base / "**" / subject_id / "**" / f"*.{ext}"), recursive=True)
            )

    candidates = sorted(set(candidates))
    if not candidates:
        return None

    # Prefer names containing eeg, then shortest path for determinism.
    candidates.sort(key=lambda p: ("eeg" not in Path(p).name.lower(), len(p), p))
    return candidates[0]


def read_raw(filepath: str):
    lower = filepath.lower()
    if lower.endswith(".set"):
        return mne.io.read_raw_eeglab(filepath, preload=True, verbose="error")
    if lower.endswith(".edf"):
        return mne.io.read_raw_edf(filepath, preload=True, verbose="error")
    if lower.endswith(".fif"):
        return mne.io.read_raw_fif(filepath, preload=True, verbose="error")
    raise ValueError(f"unsupported EEG file: {filepath}")


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    mean_psd = np.mean(psd[:, mask], axis=0)
    return float(np.trapz(mean_psd, freqs[mask]))


def spectral_features(filepath: str, max_duration_s: float = 120.0) -> dict:
    raw = read_raw(filepath)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False,
                           stim=False, exclude="bads")
    if len(picks) < 8:
        raise ValueError(f"only {len(picks)} EEG channels available")

    sfreq = float(raw.info["sfreq"])
    n_max = min(raw.n_times, int(round(max_duration_s * sfreq)))
    data = raw.get_data(picks=picks, start=0, stop=n_max)

    # Remove channel means only; do not silently impose a new reference here.
    data = data - np.mean(data, axis=1, keepdims=True)

    nperseg = min(data.shape[1], max(256, int(round(4.0 * sfreq))))
    noverlap = nperseg // 2
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=1,
    )

    total = _band_power(freqs, psd, 1.0, 45.0)
    theta = _band_power(freqs, psd, 4.0, 8.0)
    alpha = _band_power(freqs, psd, 8.0, 13.0)

    alpha_mask = (freqs >= 7.0) & (freqs <= 13.0)
    if not np.any(alpha_mask):
        peak_alpha = float("nan")
    else:
        mean_psd = np.mean(psd, axis=0)
        sub_freqs = freqs[alpha_mask]
        peak_alpha = float(sub_freqs[int(np.argmax(mean_psd[alpha_mask]))])

    return {
        "n_channels": int(len(picks)),
        "sfreq": sfreq,
        "duration_s": float(n_max / sfreq),
        "alpha_relative_power": float(alpha / total) if total > 0 else float("nan"),
        "theta_alpha_ratio": float(theta / alpha) if alpha > 0 else float("nan"),
        "peak_alpha_frequency": peak_alpha,
    }


def mannwhitney_summary(rows: List[dict], metric: str) -> dict:
    ad = np.array([r[metric] for r in rows if r["group"] == "AD" and np.isfinite(r[metric])])
    cn = np.array([r[metric] for r in rows if r["group"] == "CN" and np.isfinite(r[metric])])
    if len(ad) < 3 or len(cn) < 3:
        return {"metric": metric, "error": "too few AD/CN subjects"}
    u, p = stats.mannwhitneyu(ad, cn, alternative="two-sided")
    return {
        "metric": metric,
        "n_ad": int(len(ad)),
        "n_cn": int(len(cn)),
        "mean_ad": float(np.mean(ad)),
        "mean_cn": float(np.mean(cn)),
        "median_ad": float(np.median(ad)),
        "median_cn": float(np.median(cn)),
        "difference_ad_minus_cn": float(np.mean(ad) - np.mean(cn)),
        "mannwhitney_u": float(u),
        "p_two_sided": float(p),
    }


def spearman_subset(rows: List[dict], metric: str, group: str, target: str) -> dict:
    pairs = [
        (r[metric], r[target])
        for r in rows
        if r["group"] == group
        and r.get(target) is not None
        and np.isfinite(r[metric])
        and np.isfinite(float(r[target]))
    ]
    if len(pairs) < 5:
        return {"group": group, "metric": metric, "target": target, "n": len(pairs), "p": 1.0}
    x = np.asarray([a for a, _ in pairs], dtype=float)
    y = np.asarray([b for _, b in pairs], dtype=float)
    rho, p = stats.spearmanr(x, y)
    return {
        "group": group,
        "metric": metric,
        "target": target,
        "n": int(len(pairs)),
        "spearman_rho": float(rho),
        "p_two_sided": float(p),
    }


def cv_auc_models(rows: List[dict]) -> dict:
    """Repeated subject-wise CV; descriptive because this is the discovery cohort."""
    model_rows = [
        r for r in rows
        if r["group"] in {"AD", "CN"}
        and r.get("age") is not None
        and all(np.isfinite(r[k]) for k in (
            "dwell_gradient", "alpha_relative_power", "theta_alpha_ratio", "peak_alpha_frequency"
        ))
    ]
    if len(model_rows) < 20:
        return {"error": "too few complete AD/CN rows"}

    y = np.array([1 if r["group"] == "AD" else 0 for r in model_rows], dtype=int)
    feature_sets = {
        "A_age_plus_spectral": ["age", "alpha_relative_power", "theta_alpha_ratio", "peak_alpha_frequency"],
        "B_age_plus_dwell": ["age", "dwell_gradient"],
        "C_age_plus_spectral_plus_dwell": [
            "age", "alpha_relative_power", "theta_alpha_ratio", "peak_alpha_frequency", "dwell_gradient"
        ],
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    scores = {name: [] for name in feature_sets}

    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        for name, features in feature_sets.items():
            X = np.array([[float(r[f]) for f in features] for r in model_rows], dtype=float)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED),
            )
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) == 2:
                scores[name].append(float(roc_auc_score(y[test_idx], prob)))

    summary = {
        name: {
            "n_folds": len(vals),
            "mean_auc": float(np.mean(vals)),
            "sd_auc": float(np.std(vals, ddof=1)),
            "median_auc": float(np.median(vals)),
        }
        for name, vals in scores.items()
    }
    summary["increment_C_minus_A"] = float(
        summary["C_age_plus_spectral_plus_dwell"]["mean_auc"]
        - summary["A_age_plus_spectral"]["mean_auc"]
    )
    summary["boundary"] = (
        "Internal discovery-cohort CV only. Because dwell_gradient was discovered on ds004504, "
        "this AUC increment is not external validation."
    )
    return summary


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = [
        "subject", "group", "age", "mmse", "file", "n_channels", "sfreq", "duration_s",
        "dwell_gradient", "alpha_relative_power", "theta_alpha_ratio", "peak_alpha_frequency",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Φ-Dwell 2026 spectral-slowing internal audit")
    parser.add_argument("dataset", help="Path to OpenNeuro ds004504")
    parser.add_argument(
        "--results",
        default="Results/phidwell_alzheimer_results.json",
        help="Existing Φ-Dwell per-subject JSON used to derive dwell_gradient",
    )
    parser.add_argument("--use-derivatives", action="store_true", help="Read EEG from dataset/derivatives")
    parser.add_argument("--max-duration", type=float, default=120.0, help="Seconds per subject (default 120)")
    parser.add_argument(
        "--out",
        default="Results/phidwell_spectral_audit.json",
        help="Output JSON path; sibling CSV is also written",
    )
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        historical = json.load(handle)
    participants = parse_participants(args.dataset)

    rows: List[dict] = []
    failures: List[dict] = []

    subject_ids = sorted(set(historical) & set(participants))
    print(f"Φ-Dwell spectral audit: {len(subject_ids)} subjects in both result JSON and participants.tsv")
    print(f"EEG source: {'derivatives' if args.use_derivatives else 'dataset/raw-style'}")

    for i, sid in enumerate(subject_ids, start=1):
        p = participants[sid]
        filepath = find_subject_file(args.dataset, sid, args.use_derivatives)
        if filepath is None:
            failures.append({"subject": sid, "error": "EEG file not found"})
            print(f"[{i}/{len(subject_ids)}] {sid}: FILE NOT FOUND")
            continue
        try:
            dg = dwell_gradient(historical[sid])
            spec = spectral_features(filepath, max_duration_s=args.max_duration)
            row = {
                "subject": sid,
                "group": p["group"],
                "age": p["age"],
                "mmse": p["mmse"],
                "file": filepath,
                "dwell_gradient": dg,
                **spec,
            }
            rows.append(row)
            print(
                f"[{i}/{len(subject_ids)}] {sid} {p['group']}: "
                f"dwell={dg:+.4f} alpha_rel={spec['alpha_relative_power']:.4f} "
                f"theta/alpha={spec['theta_alpha_ratio']:.3f} PAF={spec['peak_alpha_frequency']:.2f}"
            )
        except Exception as exc:
            failures.append({"subject": sid, "file": filepath, "error": repr(exc)})
            print(f"[{i}/{len(subject_ids)}] {sid}: ERROR {exc}")

    metrics = ["dwell_gradient", "alpha_relative_power", "theta_alpha_ratio", "peak_alpha_frequency"]
    group_tests = {metric: mannwhitney_summary(rows, metric) for metric in metrics}

    within_ad_mmse = spearman_subset(rows, "dwell_gradient", "AD", "mmse")
    within_ftd_mmse = spearman_subset(rows, "dwell_gradient", "FTD", "mmse")
    within_cn_age = spearman_subset(rows, "dwell_gradient", "CN", "age")
    within_ad_age = spearman_subset(rows, "dwell_gradient", "AD", "age")

    cv_summary = cv_auc_models(rows)

    output = {
        "schema": "brain-metastability/phidwell-spectral-audit-v1",
        "status": "INTERNAL_DISCOVERY_COHORT_AUDIT_NOT_EXTERNAL_VALIDATION",
        "seed": SEED,
        "dataset": os.path.abspath(args.dataset),
        "historical_results": os.path.abspath(args.results),
        "use_derivatives": bool(args.use_derivatives),
        "max_duration_s": float(args.max_duration),
        "frozen_features": {
            "phidwell": "dwell_gradient = slope(log(1 + band_mean_dwell)) across delta,theta,alpha,beta,gamma",
            "spectral": [
                "alpha_relative_power = P(8-13)/P(1-45)",
                "theta_alpha_ratio = P(4-8)/P(8-13)",
                "peak_alpha_frequency = PSD peak in 7-13 Hz",
            ],
        },
        "n_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "group_tests_ad_vs_cn": group_tests,
        "severity_checks": {
            "within_AD_dwell_vs_MMSE": within_ad_mmse,
            "within_FTD_dwell_vs_MMSE": within_ftd_mmse,
        },
        "age_checks": {
            "within_AD_dwell_vs_age": within_ad_age,
            "within_CN_dwell_vs_age": within_cn_age,
        },
        "cross_validated_models": cv_summary,
        "interpretation_boundary": (
            "All statistics here reuse the ds004504 discovery cohort. They are internal mechanism/robustness checks. "
            "Only a frozen test on independent subjects can validate dwell_gradient."
        ),
        "subjects": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    csv_path = out_path.with_suffix(".csv")
    write_csv(csv_path, rows)

    print("\n=== AD vs CN discovery-cohort checks ===")
    for metric in metrics:
        item = group_tests[metric]
        if "error" in item:
            print(f"{metric:24s} {item['error']}")
        else:
            print(
                f"{metric:24s} AD={item['mean_ad']:.4f} CN={item['mean_cn']:.4f} "
                f"p={item['p_two_sided']:.5g}"
            )

    print("\n=== Severity sanity check ===")
    print("within AD dwell vs MMSE:", within_ad_mmse)
    print("within FTD dwell vs MMSE:", within_ftd_mmse)

    print("\n=== Subject-wise repeated CV (internal only) ===")
    if "error" in cv_summary:
        print(cv_summary["error"])
    else:
        for key in ("A_age_plus_spectral", "B_age_plus_dwell", "C_age_plus_spectral_plus_dwell"):
            print(f"{key}: mean AUC={cv_summary[key]['mean_auc']:.3f} ± {cv_summary[key]['sd_auc']:.3f}")
        print(f"C - A AUC = {cv_summary['increment_C_minus_A']:+.3f}")

    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
