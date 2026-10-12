"""Maps and charts. Each title states the finding."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from config import FIG_DIR  # noqa: E402

BLUE, INK, INK_2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#dbe8f8", "#a9c9ef", "#6fa3e3", "#2a78d6", "#174a8a"]          # one hue, light to dark
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # fixed order
NO_DATA = "#d9d8d3"
DUBLIN = (-6.48, -6.03, 53.20, 53.46)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5, "font.family": "DejaVu Sans",
    "legend.frameon": False,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _sub(ax, text):
    ax.text(0, 1.01, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


# --------------------------------------------------------------------------- maps
def _polys(geojson: Path):
    """[(guid, [ring, ...])] with rings as Nx2 arrays of lon/lat."""
    out = []
    for f in json.loads(Path(geojson).read_text())["features"]:
        g, guid = f.get("geometry") or {}, f["properties"].get("ED_GUID")
        if g.get("type") == "Polygon":
            rings = [np.asarray(g["coordinates"][0])]
        elif g.get("type") == "MultiPolygon":
            rings = [np.asarray(p[0]) for p in g["coordinates"]]
        else:
            continue
        out.append((guid, rings))
    return out


def _draw(ax, polys, colour_of, bbox=None):
    verts, cols = [], []
    for guid, rings in polys:
        c = colour_of.get(guid, NO_DATA)
        for r in rings:
            if bbox and not (r[:, 0].max() > bbox[0] and r[:, 0].min() < bbox[1]
                             and r[:, 1].max() > bbox[2] and r[:, 1].min() < bbox[3]):
                continue
            verts.append(r)
            cols.append(c)
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors=SURFACE, linewidths=0.08))
    ax.autoscale_view()
    if bbox:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1 / np.cos(np.radians(53.4)))
    ax.axis("off")


def _map(polys, colour_of, title, subtitle, legend_handles, name):
    # Ireland on the left; Dublin close-up and legend in a separate column, so nothing overlaps
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_axes([0.0, 0.0, 0.60, 0.90])
    _draw(ax, polys, colour_of)
    fig.text(0.01, 0.975, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.94, subtitle, fontsize=10, color=INK_2, va="top")
    inset = fig.add_axes([0.63, 0.47, 0.36, 0.40])
    _draw(inset, polys, colour_of, bbox=DUBLIN)
    inset.set_title("Dublin close-up", fontsize=10, fontweight="bold", loc="left")
    inset.axis("on")
    inset.set_xticks([])
    inset.set_yticks([])
    for sp in inset.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    leg_ax = fig.add_axes([0.63, 0.04, 0.36, 0.38])
    leg_ax.axis("off")
    leg_ax.legend(handles=legend_handles, loc="upper left", fontsize=9.5)
    return _save(fig, name)


def map_bad_health(ind: pd.DataFrame, geojson: Path | None) -> str | None:
    if geojson is None:
        return None
    v = ind["pct_bad_health"].dropna()
    edges = np.quantile(v, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 4)
    colour_of = {g: SEQ[b] for g, b in zip(v.index, bins)}
    handles = [Patch(facecolor=SEQ[i], label=f"{edges[i]:.1f}% to {edges[i + 1]:.1f}%") for i in range(5)]
    handles.append(Patch(facecolor=NO_DATA, label="No data"))
    return _map(_polys(geojson), colour_of,
                "Bad health is concentrated in specific communities",
                "Share of people reporting bad or very bad health, by Electoral Division (fifths), Census 2022",
                handles, "01_map_bad_health.png")


def map_profiles(prof: dict, geojson: Path | None) -> str | None:
    if geojson is None:
        return None
    t = prof["table"]
    colour_of = {g: CAT[p % len(CAT)] for g, p in prof["assign"].items()}
    handles = [Patch(facecolor=CAT[i % len(CAT)], label=f"{r['name']} ({r['bad_health']:.1f}% bad health)")
               for i, r in t.iterrows()]
    return _map(_polys(geojson), colour_of,
                f"{len(t)} types of community, with very different health",
                "Community profiles from k-means clustering of age, jobs, education and housing",
                handles, "02_map_profiles.png")


# --------------------------------------------------------------------------- charts
def effects(assoc: dict) -> str:
    c = assoc["coefs"].sort_values("effect_pp")
    fig, ax = plt.subplots(figsize=(9.5, 4))
    y = np.arange(len(c))
    ax.hlines(y, c["ci_low"], c["ci_high"], color=BLUE, lw=2.2)
    ax.plot(c["effect_pp"], y, "o", color=BLUE, ms=8)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl}  (1 SD = {sd:.1f} pp)" for lbl, sd in zip(c["label"], c["sd"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in bad-health rate (percentage points) for a 1 SD higher value, others held fixed")
    top = c.iloc[-1]
    ax.set_title(f"After allowing for age, {top['label'].split(' (')[0].lower()} has the strongest link", pad=24)
    _sub(ax, f"Weighted regression across Electoral Divisions, 95% robust confidence intervals; R² = {assoc['r2_full']:.2f}")
    return _save(fig, "03_effects.png")


def model_test(comp: dict) -> str:
    r2 = comp["r2"]
    labels = {"age_only": "Age structure only", "linear": "Linear model (all factors)",
              "forest": "Random forest (all factors)"}
    keys = ["age_only", "linear", "forest"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    vals = [r2[k] for k in keys]
    bars = ax.barh([labels[k] for k in keys], vals, color=["#9fc3ee", BLUE, BLUE], height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, max(1.0, max(vals) + 0.1))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of variation explained in counties the model never saw (out-of-sample R²)")
    best = max(keys, key=lambda k: r2[k])
    gain = r2["linear"] - r2["age_only"]
    ax.set_title(f"Social factors add {gain:.2f} to what age alone explains; "
                 f"the {'forest' if best == 'forest' else 'simpler linear model'} does best", pad=24)
    _sub(ax, f"{comp['n_splits']}-fold cross-validation holding out whole counties ({comp['n_counties']} counties)")
    return _save(fig, "04_model_test.png")


def counties(ct: pd.DataFrame, national: float) -> str:
    c = ct.sort_values("Bad or very bad health (%)")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(c.index, c["Bad or very bad health (%)"], color=BLUE, height=0.65)
    ax.axvline(national, color=INK_2, lw=1.2, ls="--")
    ax.text(national, len(c) - 0.3, f" State {national:.1f}%", color=INK_2, fontsize=9.5, va="bottom")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Bad or very bad health (%)")
    ax.set_title(f"{c.index[-1]} has the highest rate, {c.index[0]} the lowest", pad=24)
    _sub(ax, "Share reporting bad or very bad health by local authority area, Census 2022 (not age-adjusted)")
    return _save(fig, "05_counties.png")
