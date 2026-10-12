"""FAKE Census tables in the exact CSO PxStat CSV layout, plus fake map polygons, for testing.

Never publish results made from these files: every output is marked DEMO.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import RAW_DIR, TABLES

DEMO_DIR = RAW_DIR.parent / "demo"
COUNTIES = ["Carlow", "Dublin City", "South Dublin", "Fingal", "Dún Laoghaire-Rathdown", "Kildare", "Kilkenny",
            "Laois", "Longford", "Louth", "Meath", "Offaly", "Westmeath", "Wexford", "Wicklow", "Clare",
            "Cork City", "Cork County", "Kerry", "Limerick City and County", "Tipperary", "Waterford City and County",
            "Galway City", "Galway County", "Leitrim", "Mayo", "Roscommon", "Sligo", "Cavan", "Donegal", "Monaghan"]

CATS = {
    "health": ("C03760V04509", "General Health", ["Very Good", "Good", "Fair", "Bad", "Very Bad", "Not Stated"]),
    "age": ("C03737V04485", "Age", [f"Age {a}-{a + 4}" for a in range(0, 85, 5)] + ["Age 85 and over"]),
    "education": ("C03753V04502", "Highest Level of Education Completed",
                  ["No formal education", "Primary education", "Lower secondary", "Upper secondary",
                   "Technical or vocational qualification", "Advanced certificate/Completed apprenticeship",
                   "Higher certificate", "Ordinary bachelor degree or national diploma",
                   "Honours bachelor degree, professional qualification or both", "Postgraduate diploma or degree",
                   "Doctorate(Ph.D) or higher", "Not stated"]),
    "economic": ("C03742V04491", "Principle Economic Status",
                 ["At work", "Looking for first regular job", "Short term unemployed", "Long term unemployed",
                  "Student", "Looking after home/family", "Retired",
                  "Unable to work due to permanent sickness or disability", "Other"]),
    "tenure": ("C03761V04510", "Type of Occupancy",
               ["Owned with mortgage or loan", "Owned outright", "Rented from private landlord",
                "Rented from Local Authority", "Rented from voluntary/co-operative housing body",
                "Occupied free of rent", "Not stated"]),
}


def _split(total: int, probs: np.ndarray, rng) -> np.ndarray:
    return rng.multinomial(int(total), probs / probs.sum())


def make(n: int = 3420, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    county = rng.choice(COUNTIES, n)
    guid = [f"demo-{i:05d}" for i in range(n)]
    names = [f"ED {i}, {c}" for i, c in enumerate(county)]
    pop = np.clip(rng.lognormal(7.0, 0.9, n), 60, 30000).astype(int)
    dep = rng.normal(0, 1, n)            # latent deprivation
    old = rng.normal(0, 1, n)            # latent age structure
    urban = rng.random(n) < 0.35

    p65 = np.clip(0.15 + 0.05 * old - 0.03 * urban, 0.04, 0.4)
    unemp = np.clip(0.06 + 0.03 * dep + 0.01 * urban, 0.01, 0.35)
    lowedu = np.clip(0.12 + 0.05 * dep + 0.06 * old, 0.01, 0.5)
    degree = np.clip(0.33 - 0.10 * dep + 0.10 * urban, 0.03, 0.8)
    social = np.clip(0.07 + 0.06 * dep + 0.05 * urban, 0.0, 0.7)
    private = np.clip(0.12 + 0.12 * urban, 0.01, 0.6)
    bad = np.clip(0.012 + 0.06 * p65 + 0.10 * unemp + 0.03 * lowedu + 0.02 * social
                  + rng.normal(0, 0.004, n), 0.002, 0.2)

    per_ed = []
    for i in range(n):
        P = pop[i]
        fair = 0.08 + 1.5 * bad[i]
        hp = np.array([0.53 - 3 * bad[i], 0.33, fair, bad[i] * 0.8, bad[i] * 0.2, 0.03])
        ages = np.full(18, (1 - p65[i]) / 13)
        ages[13:] = p65[i] / 5
        a15 = int(P * 0.8)
        ep = np.array([0.003, lowedu[i] * 0.95, 0.12, 0.25, 0.08, 0.06, 0.05, degree[i] * 0.3,
                       degree[i] * 0.45, degree[i] * 0.23, degree[i] * 0.02, 0.06])
        work = 0.6
        cp = np.array([work, unemp[i] * work * 0.15, unemp[i] * work * 0.5, unemp[i] * work * 0.35,
                       0.1, 0.06, p65[i] * 0.9, 0.03 + 0.1 * bad[i], 0.01])
        hh = max(int(P / 2.7), 1)
        tp = np.array([0.3, 0.33, private[i], social[i] * 0.85, social[i] * 0.15, 0.02, 0.02])
        per_ed.append({
            "health": _split(P, hp, rng), "age": _split(P, ages, rng), "education": _split(a15, ep, rng),
            "economic": _split(a15, cp, rng), "tenure": _split(hh, tp, rng),
        })

    for key, table in TABLES.items():
        dim_code, dim_label, cats = CATS[key]
        rows = []
        for i in range(n):
            vals = per_ed[i][key]
            full = list(zip(cats + ["Total"], list(vals) + [int(vals.sum())]))
            if key == "tenure":
                for stat in ["Permanent private households", "Number of persons in permanent private households"]:
                    mult = 1 if stat.startswith("Permanent") else 3
                    for c, v in full:
                        rows.append(["X01", stat, "2022", "2022", c[:3], c, guid[i], names[i], "Number", v * mult])
                continue
            for sex, f in [("Males", 0.49), ("Females", 0.51), ("Both Sexes", 1.0)]:
                for c, v in full:
                    val = v if sex == "Both Sexes" else int(round(v * f))
                    rows.append(["X01", f"Demo {key}", "2022", "2022", sex[0], sex, c[:3], c,
                                 guid[i], names[i], "Number", val])
        if key == "tenure":
            cols = ["STATISTIC", "Statistic Label", "TLIST(A1)", "Census Year", dim_code, dim_label,
                    "C04167V04938", "CSO Electoral Divisions 2022", "UNIT", "VALUE"]
        else:
            cols = ["STATISTIC", "Statistic Label", "TLIST(A1)", "Census Year", "C03738V04487", "Sex",
                    dim_code, dim_label, "C04167V04938", "CSO Electoral Divisions 2022", "UNIT", "VALUE"]
        pd.DataFrame(rows, columns=cols).to_csv(DEMO_DIR / f"{table}.csv", index=False, quoting=1)

    # Fake map: a grid of squares roughly over Ireland
    side = int(np.ceil(np.sqrt(n)))
    feats = []
    for i in range(n):
        x0 = -10.4 + (i % side) * (4.4 / side)
        y0 = 51.4 + (i // side) * (4.0 / side)
        dx, dy = 4.4 / side, 4.0 / side
        feats.append({"type": "Feature", "properties": {"ED_GUID": guid[i], "ED_ENGLISH": names[i],
                                                         "COUNTY_ENGLISH": county[i]},
                      "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x0 + dx, y0], [x0 + dx, y0 + dy],
                                                                       [x0, y0 + dy], [x0, y0]]]}})
    (DEMO_DIR / "ed_boundaries_2022.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"Demo files written to {DEMO_DIR}")


if __name__ == "__main__":
    make()
