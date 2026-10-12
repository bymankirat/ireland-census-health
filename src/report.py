"""Self-contained web report written for policymakers, with the statistics behind it."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from data import INDICATORS
from stats import PREDICTORS

AUTHOR = "Mankirat Singh"
LINKS = {"Portfolio": "https://bymankirat.github.io/myportfolio/", "GitHub": "https://github.com/bymankirat",
         "LinkedIn": "https://linkedin.com/in/mankirat-singh-694440177"}


def _img(p, alt):
    if not p:
        return '<p class="muted">(Map not available: boundary download failed.)</p>'
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" alt="{html.escape(alt)}">'


def build(nat, ct, assoc, comp, prof, qc, figs, demo, min_pop_gap=1000) -> str:
    coefs = assoc["coefs"].sort_values("effect_pp", ascending=False)
    top = coefs.iloc[0]
    r2 = comp["r2"]
    pt = prof["table"]
    hi, lo = pt.iloc[0], pt.iloc[-1]
    gap = assoc["age_gap"]
    big = gap[gap["population"] >= min_pop_gap].sort_values("gap_pp", ascending=False)
    n_2pp = int((big["gap_pp"] >= 2).sum())
    max_vif = max(assoc["vif"].values())

    findings = [
        f"In Census 2022, <b>{nat['pct_bad_health']:.1f}%</b> of people reported bad or very bad health. "
        f"Across the {nat['eds']:,} Electoral Divisions this ranged from {nat['ed_p10']:.1f}% to "
        f"{nat['ed_p90']:.1f}% (10th to 90th percentile).",
        f"Age structure alone explains <b>{100 * r2['age_only']:.0f}%</b> of the variation between communities "
        f"(tested on counties the model never saw). Adding jobs, education and housing raises this to "
        f"<b>{100 * r2['linear']:.0f}%</b>.",
        f"After allowing for age, the strongest link is with <b>{top['label'].split(' (')[0].lower()}</b>: "
        f"a one-standard-deviation higher share goes with {top['effect_pp']:.2f} pp more bad health "
        f"(95% CI {top['ci_low']:.2f} to {top['ci_high']:.2f}).",
        f"Communities fall into {len(pt)} profiles. Bad health is <b>{hi['bad_health']:.1f}%</b> in "
        f"\"{hi['name']}\" areas against {lo['bad_health']:.1f}% in \"{lo['name']}\" areas.",
        f"<b>{n_2pp}</b> communities of 1,000+ people have bad-health rates at least 2 pp above what their "
        f"age profile predicts: a practical shortlist for closer local attention.",
    ]

    nat_rows = "".join(f"<tr><td>{lbl}</td><td class='num'>{nat[k]:.1f}</td></tr>"
                       for k, lbl in [("pct_bad_health", "Bad or very bad health (%)"), *INDICATORS.items()])
    corr_rows = "".join(f"<tr><td>{INDICATORS[k]}</td><td class='num'>{v:+.2f}</td></tr>"
                        for k, v in sorted(assoc["corr"].items(), key=lambda kv: -abs(kv[1])))
    coef_rows = "".join(
        f"<tr><td>{r['label']}</td><td class='num'>{r['sd']:.1f}</td><td class='num'>{r['effect_pp']:+.2f}</td>"
        f"<td class='num'>{r['ci_low']:+.2f} to {r['ci_high']:+.2f}</td><td class='num'>{assoc['vif'][r['predictor']]:.1f}</td></tr>"
        for _, r in coefs.iterrows())
    imp = sorted(comp["importance"].items(), key=lambda kv: -kv[1])
    imp_text = ", ".join(INDICATORS[k].split(" (")[0].lower() for k, _ in imp[:3])
    prof_rows = "".join(
        f"<tr><td><b>{html.escape(r['name'])}</b></td><td class='num'>{r['eds']:,}</td>"
        f"<td class='num'>{r['population'] / 1000:,.0f}k</td><td class='num'><b>{r['bad_health']:.1f}</b></td>"
        f"<td class='num'>{r['pct_65plus']:.0f}</td><td class='num'>{r['unemployment_rate']:.1f}</td>"
        f"<td class='num'>{r['pct_low_education']:.0f}</td><td class='num'>{r['pct_degree']:.0f}</td>"
        f"<td class='num'>{r['pct_social_housing']:.0f}</td><td class='num'>{r['pct_private_rent']:.0f}</td></tr>"
        for _, r in pt.iterrows())
    gap_rows = "".join(
        f"<tr><td>{html.escape(str(r['ed']))}</td><td class='num'>{r['population']:,.0f}</td>"
        f"<td class='num'>{r['observed']:.1f}</td><td class='num'>{r['expected_from_age']:.1f}</td>"
        f"<td class='num'><b>{r['gap_pp']:+.1f}</b></td></tr>" for _, r in big.head(15).iterrows())
    qc_rows = "".join(f"<tr><td class='qc-{s.lower()}'>{s}</td><td>{html.escape(c)}</td><td>{html.escape(d)}</td></tr>"
                      for s, c, d in qc["results"])

    banner = ('<div class="banner">DEMO DATA: simulated numbers for testing. Run without --demo to download '
              'the real Census 2022 data before publishing.</div>') if demo else ""
    links = " · ".join(f'<a href="{u}">{k}</a>' for k, u in LINKS.items())
    vif_note = (f" The largest variance inflation factor is {max_vif:.1f}: "
                + ("predictors overlap substantially, so treat individual coefficients with caution."
                   if max_vif > 5 else "the predictors are distinct enough to separate."))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Health and Deprivation in Ireland</title>
<style>
:root{{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e4e3df;--accent:#2a78d6;--warn-bg:#fff4e5;--warn-ink:#7a3d00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink-2:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--warn-bg:#3a2a12;--warn-ink:#ffd08a}}:root:not([data-theme="light"]) img{{background:#fcfcfb}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink-2:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--warn-bg:#3a2a12;--warn-ink:#ffd08a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:920px;margin:0 auto;padding:40px 16px 64px}}
.kicker{{color:var(--accent);font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-size:.82rem}}
h1{{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.2;margin:6px 0 8px}}h2{{font-size:1.3rem;margin:44px 0 8px}}
.byline,.muted{{color:var(--ink-2)}}a{{color:var(--accent)}}
.banner{{background:var(--warn-bg);color:var(--warn-ink);padding:12px 16px;border-radius:8px;font-weight:600;margin:16px 0}}
.note{{border-left:3px solid var(--accent);padding:6px 12px;color:var(--ink-2);font-size:.92rem;margin:14px 0}}
.findings{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px 16px 36px;margin:20px 0}}
.findings li{{margin:7px 0}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.93rem}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--ink-2);font-weight:600}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.qc-pass{{color:#008300;font-weight:700}}td.qc-warn{{color:#b06f00;font-weight:700}}td.qc-fail{{color:#e34948;font-weight:700}}
footer{{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);color:var(--ink-2);font-size:.9rem}}
</style></head><body><main>
{banner}
<div class="kicker">Independent analysis of Census 2022</div>
<h1>Health and Deprivation Across Ireland's Communities</h1>
<p class="byline">{AUTHOR} · {links}</p>
<p class="note">Independent analysis of CSO Census 2022 Small Area Population Statistics for all {nat['eds']:,}
Electoral Divisions. Not an official CSO publication. All findings describe <b>areas</b>, not individuals,
and are associations, not proof of cause.</p>

<h2>Key findings</h2>
<ul class="findings">{''.join(f'<li>{f}</li>' for f in findings)}</ul>

<h2>1. The national picture</h2>
<div class="table-wrap"><table><thead><tr><th>Indicator (Census 2022, State)</th><th class="num">Value</th></tr></thead>
<tbody>{nat_rows}</tbody></table></div>

<h2>2. Where bad health is concentrated</h2>
{_img(figs.get('map_health'), 'Map of bad health by Electoral Division')}
{_img(figs.get('counties'), 'Bad health by county')}
<p class="muted">County rates are not age-adjusted: areas with older populations will show more bad health for that reason alone.
Section 6 separates the two.</p>

<h2>3. What bad health is linked to</h2>
<p>Simple correlations across Electoral Divisions (weighted by population):</p>
<div class="table-wrap"><table><thead><tr><th>Indicator</th><th class="num">Correlation with bad health</th></tr></thead>
<tbody>{corr_rows}</tbody></table></div>
<p>Many of these factors move together, so a regression estimates each one while holding the others fixed.
Effects are per one standard deviation (SD), so they can be compared directly.{vif_note}</p>
{_img(figs.get('effects'), 'Regression effects with confidence intervals')}
<div class="table-wrap"><table><thead><tr><th>Factor</th><th class="num">1 SD (pp)</th><th class="num">Effect on bad health (pp)</th>
<th class="num">95% CI</th><th class="num">VIF</th></tr></thead><tbody>{coef_rows}</tbody></table></div>

<h2>4. How well can we predict it?</h2>
<p>To check the patterns are real and not just fitted noise, each model was trained on most counties and tested on
counties it had never seen ({comp['n_splits']}-fold cross-validation by county, so neighbouring areas cannot leak
into the test). A random forest was included to catch non-linear patterns.</p>
{_img(figs.get('model'), 'Out-of-sample R squared by model')}
<p>{'The random forest adds little over the linear model, so the simpler, explainable model is preferred.'
    if r2['forest'] - r2['linear'] < 0.03 else
    'The random forest captures patterns the linear model misses, suggesting non-linear relationships worth exploring.'}
Its most important inputs were {imp_text}.</p>

<h2>5. Community profiles</h2>
<p>Grouping Electoral Divisions by age, jobs, education and housing (k-means, k = {prof['k']} chosen by silhouette score)
gives distinct types of community. Health was <b>not</b> used to form the groups, so the differences below are a genuine test.</p>
{_img(figs.get('map_profiles'), 'Map of community profiles')}
<div class="table-wrap"><table><thead><tr><th>Profile</th><th class="num">EDs</th><th class="num">People</th>
<th class="num">Bad health %</th><th class="num">65+ %</th><th class="num">Unemp. %</th><th class="num">Low edu. %</th>
<th class="num">Degree %</th><th class="num">Social hsg %</th><th class="num">Private rent %</th></tr></thead>
<tbody>{prof_rows}</tbody></table></div>

<h2>6. Communities with more ill-health than their age profile predicts</h2>
<p>Older communities naturally report more bad health. The table compares each area with the rate expected from its
age structure alone. Areas well above expectation are where non-age factors matter most. Only areas of 1,000+
people are listed, because rates for small areas are noisy.</p>
<div class="table-wrap"><table><thead><tr><th>Electoral Division</th><th class="num">Population</th>
<th class="num">Bad health %</th><th class="num">Expected from age %</th><th class="num">Gap (pp)</th></tr></thead>
<tbody>{gap_rows}</tbody></table></div>

<h2>7. What this could mean for policy</h2>
<ul>
<li>Health need is local: rates differ several-fold between nearby communities, so county averages hide where need is greatest.</li>
<li>Age explains part of the picture, but jobs, education and housing explain substantially more, pointing to joined-up
responses across health, employment, education and housing.</li>
<li>The community profiles and the "above expected" shortlist give a transparent, reproducible starting point for
targeting, to be combined with local knowledge and health-service data.</li>
</ul>

<h2>8. Quality checks</h2>
<p>The pipeline checks the data automatically before any analysis and stops on a failure. Result: <b>{qc['summary']}</b>.</p>
<div class="table-wrap"><table><thead><tr><th>Status</th><th>Check</th><th>Detail</th></tr></thead>
<tbody>{qc_rows}</tbody></table></div>

<h2>Limitations and methods</h2>
<ul>
<li><b>Ecological fallacy:</b> results describe areas. They do not show that unemployed or less-educated individuals are
themselves in worse health.</li>
<li><b>Association, not cause:</b> poor health can also lead to unemployment and lower income, so effects run both ways.
"Unable to work due to illness or disability" was deliberately <b>not</b> used as a predictor, because it partly measures the outcome.</li>
<li><b>Self-reported health</b> on Census night (April 2022); people who did not state their health are excluded from rates.</li>
<li><b>Methods:</b> population-weighted least squares with heteroskedasticity-robust (HC1) 95% intervals; variance inflation factors;
random forest with county-grouped cross-validation and permutation importance; k-means on standardised indicators.
Python (pandas, NumPy, scikit-learn, matplotlib). Predictors: {', '.join(INDICATORS[k].split(' (')[0] for k in PREDICTORS)}.</li>
<li><b>Data:</b> CSO Census 2022 SAPS tables SAP2022T12T3ED, SAP2022T1T1AED, SAP2022T10T4ED, SAP2022T8T1ED and SAP2022T6T3ED
via the CSO PxStat API; ED boundaries from Tailte Éireann / CSO (generalised 50m). Code:
<a href="https://github.com/bymankirat/ireland-census-health">GitHub</a>.</li>
</ul>
<footer>Contains CSO data and Tailte Éireann boundary data, licensed under CC BY 4.0. Analysis and any errors are the author's own.</footer>
</main></body></html>
"""
