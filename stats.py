"""The statistics: national picture, associations, model comparison, community profiles.

Unit of analysis: the 3,420 Electoral Divisions (EDs). Every model is weighted by population,
so a tiny ED with a noisy rate counts for less than a large one. All findings are about AREAS,
not individuals (the ecological fallacy), and they are associations, not causes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold

from config import RANDOM_STATE
from data import INDICATORS, OUTCOME

# Predictors for the regression: one education measure only (low education and degree share
# are strongly correlated and would make each other's coefficient unstable).
PREDICTORS = ["pct_65plus", "unemployment_rate", "pct_low_education", "pct_social_housing", "pct_private_rent"]
PROFILE_VARS = ["pct_65plus", "unemployment_rate", "pct_low_education", "pct_degree",
                "pct_social_housing", "pct_private_rent"]


# --------------------------------------------------------------------------- helpers
def _wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def _wstd(x, w):
    m = _wmean(x, w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w)))


def _wcorr(x, y, w):
    mx, my = _wmean(x, w), _wmean(y, w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def _wr2(y, pred, w):
    return float(1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - _wmean(y, w)) ** 2))


def wls(y, X, w):
    """Weighted least squares with heteroskedasticity-robust (HC1) standard errors."""
    n, k = X.shape
    Xw = X * w[:, None]
    xtwx_inv = np.linalg.inv(X.T @ Xw)
    beta = xtwx_inv @ (Xw.T @ y)
    e = y - X @ beta
    meat = (Xw * (e ** 2)[:, None]).T @ Xw
    cov = (n / (n - k)) * xtwx_inv @ meat @ xtwx_inv
    return beta, np.sqrt(np.diag(cov)), _wr2(y, X @ beta, w), e


# --------------------------------------------------------------------------- national + county
def national(ind: pd.DataFrame) -> dict:
    s = ind.sum(numeric_only=True)
    return {
        "population": float(s["population"]),
        "eds": int(len(ind)),
        "pct_bad_health": 100 * s["bad_health_n"] / s["health_stated_n"],
        "pct_65plus": 100 * s["aged65_n"] / s["population"],
        "unemployment_rate": 100 * s["unemployed_n"] / s["labour_force_n"],
        "pct_low_education": 100 * s["low_edu_n"] / s["edu_stated_n"],
        "pct_degree": 100 * s["degree_n"] / s["edu_stated_n"],
        "pct_social_housing": 100 * s["social_rent_n"] / s["tenure_stated_n"],
        "pct_private_rent": 100 * s["private_rent_n"] / s["tenure_stated_n"],
        "ed_p10": float(ind[OUTCOME].quantile(0.10)),
        "ed_p90": float(ind[OUTCOME].quantile(0.90)),
    }


def by_county(ind: pd.DataFrame) -> pd.DataFrame:
    g = ind.groupby("county").sum(numeric_only=True)
    out = pd.DataFrame({
        "Population": g["population"],
        "Bad or very bad health (%)": 100 * g["bad_health_n"] / g["health_stated_n"],
        "Aged 65+ (%)": 100 * g["aged65_n"] / g["population"],
        "Unemployment rate (%)": 100 * g["unemployed_n"] / g["labour_force_n"],
    })
    return out.sort_values("Bad or very bad health (%)", ascending=False)


# --------------------------------------------------------------------------- associations
def associations(ind: pd.DataFrame) -> dict:
    d = ind.dropna(subset=[OUTCOME, *INDICATORS])
    w = d["health_stated_n"].to_numpy(float)
    y = d[OUTCOME].to_numpy(float)

    corr = {k: _wcorr(d[k].to_numpy(float), y, w) for k in INDICATORS}

    Z = np.column_stack([(d[k] - _wmean(d[k], w)) / _wstd(d[k], w) for k in PREDICTORS])
    X = np.column_stack([np.ones(len(d)), Z])
    beta, se, r2, _ = wls(y, X, w)
    coefs = pd.DataFrame({
        "predictor": PREDICTORS,
        "label": [INDICATORS[k] for k in PREDICTORS],
        "sd": [_wstd(d[k].to_numpy(float), w) for k in PREDICTORS],
        "effect_pp": beta[1:], "ci_low": beta[1:] - 1.96 * se[1:], "ci_high": beta[1:] + 1.96 * se[1:],
    })

    # Variance inflation factors: are predictors too entangled to separate?
    vif = {}
    for j, k in enumerate(PREDICTORS):
        others = np.column_stack([np.ones(len(d)), np.delete(Z, j, axis=1)])
        _, _, r2_j, _ = wls(Z[:, j], others, w)
        vif[k] = 1 / max(1 - r2_j, 1e-9)

    # Age-only model: how much of the variation is simply age structure?
    Xa = np.column_stack([np.ones(len(d)), d["pct_65plus"].to_numpy(float)])
    ba, _, r2_age, resid = wls(y, Xa, w)
    gap = pd.DataFrame({"ed": d["ed"], "county": d["county"], "population": d["population"],
                        "observed": y, "expected_from_age": Xa @ ba, "gap_pp": resid}, index=d.index)
    return {"corr": corr, "coefs": coefs, "r2_full": r2, "r2_age": r2_age, "vif": vif, "age_gap": gap}


# --------------------------------------------------------------------------- prediction test
def model_comparison(ind: pd.DataFrame, n_splits: int = 5) -> dict:
    """Out-of-sample R2 with whole counties held out (no spatial leakage between neighbours)."""
    d = ind.dropna(subset=[OUTCOME, *PREDICTORS])
    X = d[PREDICTORS].to_numpy(float)
    y = d[OUTCOME].to_numpy(float)
    w = d["health_stated_n"].to_numpy(float)
    groups = d["county"].to_numpy()

    preds = {"age_only": np.zeros_like(y), "linear": np.zeros_like(y), "forest": np.zeros_like(y)}
    imp = np.zeros(len(PREDICTORS))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xa_tr = np.column_stack([np.ones(len(tr)), X[tr, 0]])
        b, *_ = wls(y[tr], Xa_tr, w[tr])
        preds["age_only"][te] = np.column_stack([np.ones(len(te)), X[te, 0]]) @ b
        Xl_tr = np.column_stack([np.ones(len(tr)), X[tr]])
        b, *_ = wls(y[tr], Xl_tr, w[tr])
        preds["linear"][te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, n_jobs=1,
                                   random_state=RANDOM_STATE).fit(X[tr], y[tr], sample_weight=w[tr])
        preds["forest"][te] = rf.predict(X[te])
        pi = permutation_importance(rf, X[te], y[te], sample_weight=w[te], n_repeats=5,
                                    random_state=RANDOM_STATE, n_jobs=1)
        imp += pi.importances_mean / n_splits

    r2 = {k: _wr2(y, p, w) for k, p in preds.items()}
    return {"r2": r2, "importance": dict(zip(PREDICTORS, imp)), "n_splits": n_splits,
            "n_counties": int(pd.Series(groups).nunique())}


# --------------------------------------------------------------------------- community profiles
NAME_WORDS = {
    ("pct_65plus", 1): "Older population", ("pct_65plus", -1): "Younger population",
    ("unemployment_rate", 1): "High unemployment", ("pct_low_education", 1): "Lower education",
    ("pct_degree", 1): "Highly educated", ("pct_social_housing", 1): "Social housing",
    ("pct_private_rent", 1): "Private renters",
}


def profiles(ind: pd.DataFrame) -> dict:
    d = ind.dropna(subset=PROFILE_VARS + [OUTCOME])
    w = d["population"].to_numpy(float)
    Z = np.column_stack([(d[k] - _wmean(d[k], w)) / _wstd(d[k], w) for k in PROFILE_VARS])

    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(d), size=min(2500, len(d)), replace=False)
    scores = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z, sample_weight=w)
        scores[k] = float(silhouette_score(Z[sample], km.labels_[sample]))
    k_best = max(scores, key=scores.get)
    km = KMeans(n_clusters=k_best, n_init=20, random_state=RANDOM_STATE).fit(Z, sample_weight=w)
    labels = km.labels_

    rows = []
    for c in range(k_best):
        m = labels == c
        zc = pd.Series(Z[m].mean(axis=0), index=PROFILE_VARS)
        words = []
        for var, z in zc.sort_values(key=abs, ascending=False).items():
            sign = 1 if z > 0 else -1
            if abs(z) >= 0.5 and (var, sign) in NAME_WORDS and len(words) < 2:
                words.append(NAME_WORDS[(var, sign)])
        name = " & ".join(words) if words else "Close to average"
        sub = d[m]
        rows.append({
            "cluster": c, "name": name, "eds": int(m.sum()),
            "population": float(sub["population"].sum()),
            "bad_health": 100 * sub["bad_health_n"].sum() / sub["health_stated_n"].sum(),
            **{k: _wmean(sub[k].to_numpy(float), sub["population"].to_numpy(float)) for k in PROFILE_VARS},
        })
    table = pd.DataFrame(rows).sort_values("bad_health", ascending=False).reset_index(drop=True)
    # make names unique if two clusters got the same words
    dup = table["name"].duplicated(keep=False)
    table.loc[dup, "name"] = table.loc[dup, "name"] + " (" + (table.index[dup] + 1).astype(str) + ")"
    order = {c: i for i, c in enumerate(table["cluster"])}
    assign = pd.Series([order[c] for c in labels], index=d.index, name="profile")
    table["profile"] = range(len(table))
    return {"k": k_best, "silhouette": scores, "table": table, "assign": assign}
