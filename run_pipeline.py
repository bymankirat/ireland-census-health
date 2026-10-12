"""Census 2022 health and deprivation: download -> QC -> statistics -> maps -> report.

    python run_pipeline.py            # downloads everything automatically (first run ~1-2 minutes)
    python run_pipeline.py --demo     # simulated data for testing (outputs marked DEMO)
    python run_pipeline.py --refresh  # re-download
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import charts  # noqa: E402
import quality  # noqa: E402
import report  # noqa: E402
import stats  # noqa: E402
from config import DOCS_DIR, REPORTS_DIR, TABLES  # noqa: E402
from data import build_indicators, download_boundaries, download_tables  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("1/6  Getting Census 2022 data ...")
    if args.demo:
        from make_demo_data import DEMO_DIR, make
        make()
        paths = {k: DEMO_DIR / f"{t}.csv" for k, t in TABLES.items()}
        geo = DEMO_DIR / "ed_boundaries_2022.geojson"
    else:
        try:
            paths = download_tables(force=args.refresh)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"\n[!] Download failed: {e}\n    Check your internet connection and run again.")
        geo = download_boundaries(force=args.refresh)

    print("2/6  Building indicators and checking quality ...")
    ind, raw = build_indicators(paths)
    qc = quality.run_checks(ind, raw, geo)
    print(f"     {qc['summary']}  (details: reports/qc_log.txt)")
    for s, c, d in qc["results"]:
        if s != "PASS":
            print(f"       [{s}] {c}: {d}")
    if not qc["passed"]:
        sys.exit("\n[!] Quality checks failed, so no report was produced. See reports/qc_log.txt.")

    print("3/6  Statistics ...")
    nat, ct = stats.national(ind), stats.by_county(ind)
    assoc = stats.associations(ind)
    print("4/6  Model comparison (county-grouped cross-validation) ...")
    comp = stats.model_comparison(ind)
    prof = stats.profiles(ind)

    print("5/6  Maps and charts ...")
    figs = {
        "map_health": charts.map_bad_health(ind, geo), "map_profiles": charts.map_profiles(prof, geo),
        "effects": charts.effects(assoc), "model": charts.model_test(comp),
        "counties": charts.counties(ct, nat["pct_bad_health"]),
    }

    print("6/6  Writing the report ...")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(report.build(nat, ct, assoc, comp, prof, qc, figs, args.demo),
                                         encoding="utf-8")
    REPORTS_DIR.mkdir(exist_ok=True)
    out = ind.join(prof["assign"]).join(assoc["age_gap"][["expected_from_age", "gap_pp"]])
    out.to_csv(REPORTS_DIR / "ed_indicators.csv", float_format="%.3f")
    ct.to_csv(REPORTS_DIR / "county_table.csv", float_format="%.2f")
    assoc["coefs"].to_csv(REPORTS_DIR / "regression.csv", index=False, float_format="%.4f")
    prof["table"].to_csv(REPORTS_DIR / "profiles.csv", index=False, float_format="%.2f")
    (REPORTS_DIR / "results.json").write_text(json.dumps(
        {"demo": args.demo, "national": nat, "r2_out_of_sample": comp["r2"], "r2_in_sample_full": assoc["r2_full"],
         "vif": assoc["vif"], "k": prof["k"], "silhouette": prof["silhouette"]}, indent=2, default=float))

    top = assoc["coefs"].sort_values("effect_pp", ascending=False).iloc[0]
    cv = (f"**Health and Deprivation Across Ireland's Communities (Python, pandas, scikit-learn):** automated "
          f"pipeline pulling five Census 2022 tables for all {nat['eds']:,} Electoral Divisions from the CSO API, "
          f"with automatic quality checks; population-weighted regression with robust confidence intervals, "
          f"random forest validated on held-out counties (out-of-sample R² {comp['r2']['age_only']:.2f} for age "
          f"alone vs {max(comp['r2']['linear'], comp['r2']['forest']):.2f} with social factors), k-means community "
          f"profiles and ED-level maps; findings written for policymakers.")
    (REPORTS_DIR / "summary.md").write_text(("> DEMO DATA - do not publish.\n\n" if args.demo else "") + cv + "\n")

    r2 = comp["r2"]
    print("\n================ RESULTS ================" + ("  (DEMO DATA - NOT REAL)" if args.demo else ""))
    print(f"Bad or very bad health, State: {nat['pct_bad_health']:.2f}%   (EDs p10-p90: "
          f"{nat['ed_p10']:.1f}-{nat['ed_p90']:.1f}%)")
    print(f"Out-of-sample R2: age only {r2['age_only']:.2f} | linear {r2['linear']:.2f} | forest {r2['forest']:.2f}")
    print(f"Strongest link after age: {top['label']} ({top['effect_pp']:+.2f} pp per SD)")
    print(f"Community profiles: {prof['k']}")
    for _, r in prof["table"].iterrows():
        print(f"   {r['name']:<45} {r['bad_health']:.1f}% bad health, {r['eds']} EDs")
    print(f"\nReport: {DOCS_DIR / 'index.html'}\nCV line: {REPORTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
