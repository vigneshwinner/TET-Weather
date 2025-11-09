import requests
import io, zipfile, csv
from datetime import datetime, timedelta

# -----------------------------
# 0) CONFIG: put your creds here
# -----------------------------
USERNAME = "jacobrojas@utexas.edu"
PASSWORD = "Futur31nFin@nce"
SUBSCRIPTION_KEY = "69209701ce6943afb27374b59cea4bca"

# Product & artifact you want
TARGET_EMIL_ID = "NP6-345-CD"  # Actual System Load by Weather Zone
TARGET_ARTIFACT_NAME = "act_sys_load_by_wzn"  # the endpoint path

# -----------------------------
# 1) AUTHENTICATE (get id_token)
# -----------------------------
AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)

auth_form = {
    "username": USERNAME,
    "password": PASSWORD,
    "grant_type": "password",
    "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
    "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
    "response_type": "id_token",
}

auth_resp = requests.post(AUTH_URL, data=auth_form)
auth_resp.raise_for_status()
id_token = auth_resp.json()["id_token"]

headers = {
    "Authorization": f"Bearer {id_token}",
    "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
}

# -----------------------------
# 2) LIST DATASETS (optional)
# -----------------------------
products_url = "https://api.ercot.com/api/public-reports"
prod_resp = requests.get(products_url, headers=headers)
prod_resp.raise_for_status()
products = prod_resp.json()["_embedded"]["products"]

print("\nFirst 10 datasets available:")
for p in products[:10]:
    print(f"  {p['emilId']} - {p['name']}")

# -----------------------------
# 3) FIND YOUR ARTIFACT URL
# -----------------------------
artifact_url = None
for p in products:
    if p["emilId"] == TARGET_EMIL_ID:
        # print artifacts so you can see choices
        print(f"\nSelected dataset: {p['emilId']} - {p['name']}\nArtifacts:")
        for a in p.get("artifacts", []):
            href = a["_links"]["endpoint"]["href"]
            print("  ", href)
            if href.rstrip("/").endswith("/" + TARGET_ARTIFACT_NAME):
                artifact_url = href

# Fallback: if we didn't match by name, just take the first artifact
if artifact_url is None:
    for p in products:
        if p["emilId"] == TARGET_EMIL_ID and p.get("artifacts"):
            artifact_url = p["artifacts"][0]["_links"]["endpoint"]["href"]
            print("\n(Info) Exact artifact name not found; using first artifact:")
            print("  ", artifact_url)
            break

if not artifact_url:
    raise RuntimeError(f"Could not find artifacts for EMIL {TARGET_EMIL_ID}")

# -----------------------------
# 4) PULL LAST 60 DAYS (paged)
# -----------------------------
today = datetime.today().date()
start_date = today - timedelta(days=60)
print(f"\nPulling data from {start_date} to {today}")

params = {
    "operatingDayFrom": start_date.strftime("%Y-%m-%d"),
    "operatingDayTo": today.strftime("%Y-%m-%d"),
    "page": 1,
    "size": 1000,
}

all_rows = []
columns = None

while True:
    r = requests.get(artifact_url, headers=headers, params=params)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()

    # ZIP case (rare for this endpoint, but handled)
    if "zip" in ctype or artifact_url.endswith(".zip"):
        print("ZIP file returned — extracting...")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("ercot_data")
        print("Extracted files into ercot_data/ (you can stop here).")
        all_rows = []  # we won't CSV-write below since files are already extracted
        break

    # JSON case: expected for act_sys_load_by_wzn
    js = r.json()

    # columns from schema, once
    if columns is None:
        fields = js.get("fields", [])
        columns = [f["name"] for f in fields] if fields else None

    data_rows = js.get("data", [])
    all_rows.extend(data_rows)

    meta = js.get("_meta", {}) or {}
    total_pages = meta.get("totalPages", 1)
    print(f"Fetched page {params['page']} of {total_pages} (rows this page: {len(data_rows)})")

    if params["page"] >= total_pages:
        break
    params["page"] += 1

# -----------------------------
# 5) SAVE TO CSV
# -----------------------------
if all_rows:
    out = "ercot_last_60_weather_zone.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if columns:
            w.writerow(columns)
        w.writerows(all_rows)
    print(f"\n✅ Saved {len(all_rows):,} rows to {out}")
else:
    print("\nℹ️ No rows written (either ZIP was extracted or no data in range).")