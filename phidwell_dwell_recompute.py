#!/usr/bin/env python3
"""
Recompute the frozen Φ-Dwell per-band dwell measurements on raw or derivative
EEG while discarding the legacy exploratory metrics.

This deliberately reuses the historical signal transform in
`phidwell_alzheimers.py` so the robustness question is about preprocessing,
not about silently changing the feature.

Usage:
    python phidwell_dwell_recompute.py "path/to/ds004504" \
        --out Results/phidwell_dwell_raw_recompute.json

    python phidwell_dwell_recompute.py "path/to/ds004504" \
        --use-derivatives \
        --out Results/phidwell_dwell_derivatives.json

The output has `band_mean_dwell` at subject top level so it can be passed
straight into `phidwell_spectral_audit.py --results ...`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from phidwell_alzheimers import (
    BAND_NAMES,
    ELECTRODE_POS_19,
    analyze_subject,
    build_graph_laplacian,
    find_eeg_files,
)


def _as_float(value):
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"n/a", "na", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_participants_fixed(dataset: str) -> dict:
    """Read the actual BIDS field names (`Age`, `Group`, `MMSE`, `Gender`)."""
    path = Path(dataset) / "participants.tsv"
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = str(row.get("participant_id", "")).strip()
            if not sid:
                continue
            out[sid] = {
                "group": str(row.get("Group", row.get("group", ""))).strip(),
                "age": _as_float(row.get("Age", row.get("age"))),
                "mmse": _as_float(row.get("MMSE", row.get("mmse"))),
                "gender": str(row.get("Gender", row.get("gender", row.get("sex", "")))).strip(),
            }
    return out


def frozen_gradient(band_mean_dwell: dict) -> float:
    values = [np.log(float(band_mean_dwell[b]) + 1.0) for b in BAND_NAMES]
    slope, _, _, _, _ = stats.linregress(np.arange(len(BAND_NAMES), dtype=float), values)
    return float(slope)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute frozen Φ-Dwell dwell feature")
    parser.add_argument("dataset", help="Path to OpenNeuro ds004504")
    parser.add_argument("--use-derivatives", action="store_true")
    parser.add_argument("--max-duration", type=int, default=120)
    parser.add_argument("--out", required=True, help="Output JSON")
    args = parser.parse_args()

    participants = parse_participants_fixed(args.dataset)
    files = find_eeg_files(args.dataset, args.use_derivatives)
    graph_names, _, eigenvecs, _ = build_graph_laplacian(ELECTRODE_POS_19)

    print(f"Frozen dwell recompute: {len(files)} EEG files")
    print(f"Source: {'derivatives' if args.use_derivatives else 'raw-style'}")

    output = {}
    failures = []
    for i, (sid, path) in enumerate(sorted(files.items()), 1):
        metrics = analyze_subject(
            path,
            graph_names,
            eigenvecs,
            word_step_ms=25,
            max_duration_s=args.max_duration,
        )
        if metrics is None:
            failures.append({"subject": sid, "file": path})
            print(f"[{i}/{len(files)}] {sid}: FAILED")
            continue

        dwell = {b: float(metrics["band_mean_dwell"][b]) for b in BAND_NAMES}
        gradient = frozen_gradient(dwell)
        p = participants.get(sid, {})
        output[sid] = {
            "group": {"A": "AD", "F": "FTD", "C": "CN"}.get(p.get("group"), p.get("group")),
            "mmse": p.get("mmse"),
            "age": p.get("age"),
            "gender": p.get("gender"),
            "band_mean_dwell": dwell,
            "dwell_gradient": gradient,
            "n_channels": int(metrics["n_channels"]),
            "duration_s": float(metrics["duration_s"]),
            "source_file": path,
        }
        print(f"[{i}/{len(files)}] {sid}: dwell_gradient={gradient:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    meta = {
        "status": "ROBUSTNESS_RECOMPUTE_NOT_EXTERNAL_VALIDATION",
        "use_derivatives": bool(args.use_derivatives),
        "max_duration_s": int(args.max_duration),
        "n_ok": len(output),
        "n_failed": len(failures),
        "failures": failures,
        "feature": "frozen dwell_gradient from unchanged historical spatial-phase/dwell transform",
    }
    meta_path = out.with_name(out.stem + "_meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
