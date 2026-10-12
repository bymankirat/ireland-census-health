# Finish this project in 30 minutes

Everything downloads automatically. No manual data steps.

## 1. Run it (5 min)

Unzip the folder, then in Terminal:

```
cd ~/MSc_Thesis/THESIS_PAPER_DATA_PDF/03_Gut_Microbiota_Dysbiosis/ireland-census-health
python run_pipeline.py
open docs/index.html
```

(If you unzipped it elsewhere, type `cd ` and drag the folder into Terminal.) The first run downloads
about 70 MB and takes 1-2 minutes.

Check the report: **no orange DEMO banner**, the map looks like Ireland, and the QC section says PASSED.
Paste the RESULTS block from Terminal to Claude to check before publishing.

If the map is missing, the boundary download failed: the rest of the report still works. Run
`python run_pipeline.py --refresh` later to try again.

## 2. Put it on GitHub (15 min)

1. github.com → **+ → New repository** → name exactly `ireland-census-health`, **Public**, no README.
2. **uploading an existing file** → in **Chrome**, drag in `docs`, `reports`, `src` (folders, one at a time
   is fine) and `README.md`, `requirements.txt`, `run_pipeline.py`, `RUN_IN_30_MIN.md`. Don't upload `data`.
3. **Commit changes**, then **Settings → Pages → Deploy from a branch → main → /docs → Save**.
4. Live in about 3 minutes at `https://bymankirat.github.io/ireland-census-health/`.

## 3. CV

`cat reports/summary.md` gives your CV line with the real numbers. Send it to Claude to add to your CV.

## Interview talking points (CSO)

- **Why Electoral Divisions, not counties?** 3,420 areas instead of 26 gives enough data to model and
  validate honestly; county averages hide local need.
- **How did you validate?** Whole counties held out in cross-validation, so neighbours can't leak into the test.
- **Why weight by population?** Rates for tiny areas are noisy.
- **What can't you conclude?** Nothing about individuals (ecological fallacy), and not causation: ill
  health also causes unemployment. That's why "unable to work due to illness" was left out.
- **Why was health not used in the clustering?** So the health differences between profiles are a real test, not circular.
- **Quality control:** 13 automatic checks, and the run stops on any failure.
