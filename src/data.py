"""Download Census 2022 ED tables and boundaries, then build one row of indicators per ED."""

from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from config import API, BOUNDARY_FILE, BOUNDARY_URL, RAW_DIR, TABLES

STANDARD_COLS = {"STATISTIC", "Statistic Label", "TLIST(A1)", "Census Year", "Sex", "UNIT", "VALUE"}


# --------------------------------------------------------------------------- download
def _fetch(url: str) -> bytes:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()
    except Exception:  # noqa: BLE001  some Python installs lack SSL certificates; curl usually works
        return subprocess.run(["curl", "-sSL", "--max-time", "300", url],
                              capture_output=True, check=True).stdout


def download_tables(force: bool = False, folder: Path = RAW_DIR) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, table in TABLES.items():
        p = folder / f"{table}.csv"
        if force or not p.exists():
            print(f"     downloading {table} ({key}) ...")
            data = _fetch(API.format(table=table))
            if b"STATISTIC" not in data[:500]:
                raise ValueError(f"Unexpected response for {table}")
            p.write_bytes(data)
        paths[key] = p
    return paths


def download_boundaries(force: bool = False) -> Path | None:
    """ED polygons as GeoJSON (WGS84), fetched in pages of 2,000. Returns None if unavailable."""
    if BOUNDARY_FILE.exists() and not force:
        return BOUNDARY_FILE
    features, offset = [], 0
    try:
        while True:
            q = urllib.parse.urlencode({
                "where": "1=1", "outFields": "ED_GUID,ED_ENGLISH,COUNTY_ENGLISH", "outSR": 4326,
                "f": "geojson", "resultOffset": offset, "resultRecordCount": 2000,
                "geometryPrecision": 4, "maxAllowableOffset": 0.0015,
            })
            page = json.loads(_fetch(f"{BOUNDARY_URL}?{q}"))
            got = page.get("features", [])
            features += got
            if len(got) < 2000:
                break
            offset += 2000
    except Exception as e:  # noqa: BLE001
        print(f"     [!] Could not download ED boundaries ({e}); maps will be skipped.")
        return None
    BOUNDARY_FILE.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return BOUNDARY_FILE


# --------------------------------------------------------------------------- read
def read_pxstat(path: Path, statistic: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Wide table: one row per ED (index = ED GUID), one column per category, both sexes."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    geo_label = next(c for c in raw.columns if "Electoral Division" in c)
    geo_code = raw.columns[raw.columns.get_loc(geo_label) - 1]
    cat_col = next(c for c in raw.columns
                   if c not in STANDARD_COLS | {geo_label, geo_code} and not c.startswith("C0"))

    df = raw
    if "Sex" in df.columns:
        df = df[df["Sex"].str.strip().str.lower() == "both sexes"]
    if statistic:  # exact label, e.g. households rather than persons in households
        df = df[df["Statistic Label"].str.strip().str.lower() == statistic.lower()]
    df = df[~df[geo_label].str.strip().str.lower().isin({"ireland", "state"})]

    df = df.assign(value=pd.to_numeric(df["VALUE"], errors="coerce"),
                   cat=df[cat_col].str.strip().str.lower())
    wide = df.pivot_table(index=geo_code, columns="cat", values="value", aggfunc="first")
    names = df.drop_duplicates(geo_code).set_index(geo_code)[geo_label].str.strip()
    return wide, names


def _get(w: pd.DataFrame, *labels: str) -> pd.Series:
    cols = [c for c in w.columns if c in {lbl.lower() for lbl in labels}]
    if len(cols) != len(labels):
        raise KeyError(f"Expected categories {labels}, found {sorted(w.columns)}")
    return w[cols].sum(axis=1, min_count=1)


# --------------------------------------------------------------------------- indicators
def build_indicators(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    health, names = read_pxstat(paths["health"])
    age, _ = read_pxstat(paths["age"])
    edu, _ = read_pxstat(paths["education"])
    econ, _ = read_pxstat(paths["economic"])
    ten, _ = read_pxstat(paths["tenure"], statistic="Permanent private households")

    h_stated = _get(health, "Total") - _get(health, "Not Stated")
    e_stated = _get(edu, "Total") - _get(edu, "Not stated")
    t_stated = _get(ten, "Total") - _get(ten, "Not stated")
    unemployed = _get(econ, "Looking for first regular job", "Short term unemployed", "Long term unemployed")
    labour_force = unemployed + _get(econ, "At work")

    ind = pd.DataFrame({
        "ed": names,
        "population": _get(age, "Total"),
        "bad_health_n": _get(health, "Bad", "Very Bad"),
        "health_stated_n": h_stated,
        "aged65_n": _get(age, "Age 65-69", "Age 70-74", "Age 75-79", "Age 80-84", "Age 85 and over"),
        "unemployed_n": unemployed,
        "labour_force_n": labour_force,
        "low_edu_n": _get(edu, "No formal education", "Primary education"),
        "degree_n": _get(edu, "Ordinary bachelor degree or national diploma",
                         "Honours bachelor degree, professional qualification or both",
                         "Postgraduate diploma or degree", "Doctorate(Ph.D) or higher"),
        "edu_stated_n": e_stated,
        "social_rent_n": _get(ten, "Rented from Local Authority",
                              "Rented from voluntary/co-operative housing body"),
        "private_rent_n": _get(ten, "Rented from private landlord"),
        "tenure_stated_n": t_stated,
        "health_not_stated_n": _get(health, "Not Stated"),
        "health_total_n": _get(health, "Total"),
    })
    ind["county"] = ind["ed"].str.rsplit(",", n=1).str[-1].str.strip()
    ind["pct_bad_health"] = 100 * ind["bad_health_n"] / ind["health_stated_n"]
    ind["pct_65plus"] = 100 * ind["aged65_n"] / ind["population"]
    ind["unemployment_rate"] = 100 * ind["unemployed_n"] / ind["labour_force_n"]
    ind["pct_low_education"] = 100 * ind["low_edu_n"] / ind["edu_stated_n"]
    ind["pct_degree"] = 100 * ind["degree_n"] / ind["edu_stated_n"]
    ind["pct_social_housing"] = 100 * ind["social_rent_n"] / ind["tenure_stated_n"]
    ind["pct_private_rent"] = 100 * ind["private_rent_n"] / ind["tenure_stated_n"]
    ind.index.name = "ed_guid"

    raw_tables = {"health": health, "age": age, "education": edu, "economic": econ, "tenure": ten}
    return ind, raw_tables


INDICATORS = {
    "pct_65plus": "Aged 65 and over (%)",
    "unemployment_rate": "Unemployment rate (%)",
    "pct_low_education": "Primary education or less (%)",
    "pct_degree": "Degree or higher (%)",
    "pct_social_housing": "Households in social housing (%)",
    "pct_private_rent": "Households renting privately (%)",
}
OUTCOME = "pct_bad_health"
OUTCOME_LABEL = "Bad or very bad health (%)"
