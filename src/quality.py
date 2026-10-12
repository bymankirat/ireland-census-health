"""Quality control: automatic checks before any analysis. A FAIL stops the pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import EXPECTED_EDS, REPORTS_DIR, SMALL_POP
from data import INDICATORS, OUTCOME


def run_checks(ind: pd.DataFrame, raw: dict[str, pd.DataFrame], boundaries: Path | None) -> dict:
    res: list[tuple[str, str, str]] = []
    add = lambda s, c, d: res.append((s, c, d))  # noqa: E731

    # 1. Coverage: every table has every ED, and the same EDs
    counts = {k: len(v) for k, v in raw.items()}
    lo = min(counts.values())
    add("FAIL" if lo < 0.9 * EXPECTED_EDS else "PASS" if lo == EXPECTED_EDS else "WARN",
        f"{EXPECTED_EDS:,} Electoral Divisions in every table",
        ", ".join(f"{k}: {n:,}" for k, n in counts.items()))
    sets = [set(v.index) for v in raw.values()]
    union, common = set.union(*sets), set.intersection(*sets)
    mismatch = len(union) - len(common)
    add("FAIL" if mismatch > 0.01 * len(union) else "WARN" if mismatch else "PASS",
        "Same EDs in all five tables", f"{mismatch} EDs missing from at least one table")

    # 2. Categories add up to the total in every table (Census counts are exact)
    for name, w in raw.items():
        tot = next((c for c in w.columns if c == "total"), None)
        if tot is None:
            add("FAIL", f"{name}: has a Total column", "no Total column found")
            continue
        parts = w.drop(columns=[tot]).sum(axis=1)
        off = ((parts - w[tot]).abs() > 0.005 * w[tot].clip(lower=1)).mean()
        add("PASS" if off == 0 else "WARN" if off < 0.01 else "FAIL",
            f"{name}: categories add up to total", f"{100 * off:.2f}% of EDs differ by more than 0.5%")

    # 3. Missing and impossible values
    key = ind[[OUTCOME, *INDICATORS]]
    n_na = int(key.isna().sum().sum())
    add("FAIL" if n_na > 0.01 * key.size else "WARN" if n_na else "PASS",
        "No missing indicator values", f"{n_na} missing out of {key.size:,}")
    bad = int(((key < 0) | (key > 100)).sum().sum())
    add("FAIL" if bad else "PASS", "All percentages between 0 and 100", f"{bad} out of range")

    # 4. Consistency between tables: health table covers the whole population
    pop_gap = (ind["health_total_n"] - ind["population"]).abs() / ind["population"].clip(lower=1)
    worst = float(pop_gap.max() * 100)
    add("PASS" if worst <= 1 else "WARN", "Health table population matches age table",
        f"largest difference {worst:.2f}%")

    # 5. Data quality of the outcome itself
    ns = 100 * ind["health_not_stated_n"].sum() / ind["health_total_n"].sum()
    high_ns = int((100 * ind["health_not_stated_n"] / ind["health_total_n"] > 15).sum())
    add("WARN" if ns > 5 or high_ns > 50 else "PASS", "Health question answered",
        f"{ns:.1f}% did not state their health nationally; {high_ns} EDs above 15%")

    # 6. Small areas give noisy rates: flag, and the models weight by population
    small = int((ind["population"] < SMALL_POP).sum())
    add("WARN" if small else "PASS", f"EDs with population under {SMALL_POP}",
        f"{small} EDs (kept, but weighted by population in all models)")

    # 7. Boundaries for the maps
    if boundaries is None:
        add("WARN", "ED boundaries for maps", "not available; maps skipped")
    else:
        guids = {f["properties"].get("ED_GUID") for f in json.loads(boundaries.read_text())["features"]}
        matched = len(set(ind.index) & guids) / max(len(ind), 1)
        add("PASS" if matched >= 0.99 else "WARN", "EDs matched to map boundaries", f"{100 * matched:.1f}% matched")

    n_fail = sum(s == "FAIL" for s, *_ in res)
    n_warn = sum(s == "WARN" for s, *_ in res)
    summary = f"QC {'FAILED' if n_fail else 'PASSED'}: {len(res) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failures"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "qc_log.txt").write_text(
        "\n".join([f"Quality control log - {datetime.now():%Y-%m-%d %H:%M}", summary, ""]
                  + [f"[{s}] {c}: {d}" for s, c, d in res]) + "\n")
    return {"passed": n_fail == 0, "summary": summary, "results": res}
