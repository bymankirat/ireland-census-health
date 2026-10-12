"""Project settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"

API = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{table}/CSV/1.0/en"

# Census 2022 Small Area Population Statistics, Electoral Division level (3,420 EDs)
TABLES = {
    "health": "SAP2022T12T3ED",     # General health of population, by sex
    "age": "SAP2022T1T1AED",        # Population by age group and sex
    "education": "SAP2022T10T4ED",  # Highest level of education completed (15+), by sex
    "economic": "SAP2022T8T1ED",    # Principal economic status (15+), by sex
    "tenure": "SAP2022T6T3ED",      # Permanent private households by type of occupancy
}

# ED boundaries (Tailte Eireann / CSO, generalised 50m), joined on ED_GUID
BOUNDARY_URL = ("https://services-eu1.arcgis.com/BuS9rtTsYEV5C0xh/ArcGIS/rest/services/"
                "CSO_ELECTORAL_DIVISIONS_2022_Genralised_50m_view/FeatureServer/4/query")
BOUNDARY_FILE = RAW_DIR / "ed_boundaries_2022.geojson"

EXPECTED_EDS = 3420
SMALL_POP = 200          # EDs below this population are kept but flagged: their rates are noisy
RANDOM_STATE = 42
