import os
import time
import math
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import pandas as pd

# ---------------------------------------------------
# User configuration
# ---------------------------------------------------

USERNAME = "jacobrojas@utexas.edu"
PASSWORD = "Futur31nFin@nce"
SUBSCRIPTION_KEY = "0ae6c9923c534824909b7fb7d1a0bb40"

# ---- Window controls ----
# How long is the window? (in days)  -> ~6 months
WINDOW_LENGTH_DAYS = 180

# How far back the END of the window is, from "today" (in days)
# 0   -> [today - 180d, today]
# 180 -> [today - 360d, today - 180d]
# 360 -> [today - 540d, today - 360d], etc.
WINDOW_OFFSET_DAYS = 0

# File names (saved directly in working directory)
SOLAR_FILENAME = "NP4_745_SOLAR_GEN_ONLY_WINDOW.csv"
WIND_FILENAME  = "NP4_742_WIND_GEN_ONLY_WINDOW.csv"

# Each ERCOT report covers a 48-hour rolling window
WINDOW_HOURS = 48

# To cover our window, we must go back WINDOW_LENGTH_DAYS + WINDOW_OFFSET_DAYS
TOTAL_DAYS_FROM_NOW = WINDOW_LENGTH_DAYS + WINDOW_OFFSET_DAYS

# Max number of 48-hour reports we *try* to use
NUM_REPORTS = math.ceil(TOTAL_DAYS_FROM_NOW * 24 / WINDOW_HOURS)

# How far back to search in the archive (a bit more than we strictly need)
ARCHIVE_LOOKBACK_DAYS = TOTAL_DAYS_FROM_NOW + 10

# Base pause between downloads to be polite
SLEEP_SECONDS = 2

# Maximum number of retries on 429 / transient HTTP errors
MAX_RETRIES = 8

# ---------------------------------------------------
# ERCOT API constants
# ---------------------------------------------------

AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)

BASE_URL = "https://api.ercot.com/api/public-reports"

REPORT_CONFIGS = [
    {
        "label": "SOLAR",
        "emil_id": "np4-745-cd",  # Solar Hourly Averaged Values by Geo Region
        "filename": SOLAR_FILENAME,
    },
    {
        "label": "WIND",
        "emil_id": "np4-742-cd",  # Wind Hourly Averaged Values by Geo Region
        "filename": WIND_FILENAME,
    },
]

# Timestamp format ERCOT expects for query params
TS_FMT = "%Y-%m-%dT%H:%M:%S"

CENTRAL_TZ = ZoneInfo("America/Chicago")

# ---------------------------------------------------
# Authentication
# ---------------------------------------------------

def get_id_token(username: str, password: str, subscription_key: str) -> str:
    """
    Authenticate against ERCOT Public API and return the id_token.
    """
    payload = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "response_type": "id_token",
        "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
        "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
    }

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    resp = requests.post(AUTH_URL, data=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    if "id_token" not in data:
        raise RuntimeError(f"Authentication failed, no id_token in response: {data}")

    return data["id_token"]


def make_headers(id_token: str, subscription_key: str) -> dict:
    return {
        "Authorization": f"Bearer {id_token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }

# ---------------------------------------------------
# Archive access helpers (with retry)
# ---------------------------------------------------

def parse_post_datetime(dt_str: str) -> datetime:
    """
    Parse ERCOT postDatetime string into a timezone-aware datetime in Central time.
    """
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CENTRAL_TZ)
    else:
        dt = dt.astimezone(CENTRAL_TZ)
    return dt


def fetch_archives_for_window(
    headers: dict,
    emil_id: str,
    start_dt: datetime,
    end_dt: datetime,
    size: int = 1000,
    max_retries: int = MAX_RETRIES,
):
    """
    Hit the historical archive endpoint for the given EMIL ID between start_dt and end_dt,
    with retry/backoff on 429 and transient HTTP errors.
    """
    url = f"{BASE_URL}/archive/{emil_id}"

    def _get_page(page: int):
        params = {
            "postDatetimeFrom": start_dt.strftime(TS_FMT),
            "postDatetimeTo": end_dt.strftime(TS_FMT),
            "size": size,
            "page": page,
        }

        for attempt in range(max_retries):
            resp = requests.get(url, headers=headers, params=params)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, ...
                print(
                    f"⚠️  429 Too Many Requests for {url} (page {page}). "
                    f"Waiting {wait} seconds (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(wait)
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                wait = 5 * (attempt + 1)
                print(
                    f"⚠️  HTTP error {resp.status_code} on {url} (page {page}): {e}. "
                    f"Waiting {wait} seconds before retry..."
                )
                time.sleep(wait)
                continue

            return resp.json()

        raise RuntimeError(
            f"Failed to fetch archive page {page} for {emil_id} after {max_retries} retries."
        )

    # First page
    first_data = _get_page(page=1)
    archives = first_data.get("archives", [])

    meta = first_data.get("_meta", {})
    total_pages = meta.get("totalPages", 1)

    # Remaining pages
    for page in range(2, total_pages + 1):
        data = _get_page(page=page)
        archives.extend(data.get("archives", []))

    return archives


def pick_reports_every_48_hours(archives, now_dt: datetime, num_reports: int, hour_step: int):
    """
    From a list of archives, pick up to num_reports reports such that:
      - report[0] is the latest available;
      - report[k] is the latest with postDatetime <= now_dt - k * hour_step hours.

    If we run out of history (no report <= cutoff), we STOP and return
    whatever we have instead of raising an error.
    """
    enriched = []
    for a in archives:
        pd_str = a.get("postDatetime")
        if not pd_str:
            continue
        try:
            pd_dt = parse_post_datetime(pd_str)
        except Exception:
            continue
        enriched.append((a, pd_dt))

    enriched.sort(key=lambda x: x[1], reverse=True)

    if not enriched:
        raise RuntimeError("No archives found with valid postDatetime values.")

    earliest_dt = enriched[-1][1]

    def latest_as_of(cutoff: datetime):
        for a, pd_dt in enriched:
            if pd_dt <= cutoff:
                return a, pd_dt
        return None

    selected = []
    for k in range(num_reports):
        cutoff = now_dt - timedelta(hours=hour_step * k)
        choice = latest_as_of(cutoff)
        if choice is None:
            print(
                f"Reached beginning of history at cutoff {cutoff.isoformat()} "
                f"(earliest available postDatetime is {earliest_dt.isoformat()}). "
                f"Using only {len(selected)} reports instead of requested {num_reports}."
            )
            break
    selected.append(choice)

    return selected

# ---------------------------------------------------
# Cleaning helpers (robust SYSTEM_WIDE_GEN)
# ---------------------------------------------------

def clean_to_generation_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
      - DELIVERY_DATE
      - HOUR_ENDING
      - SYSTEM_WIDE_GEN  (if missing, compute as sum of GEN_* columns)
      - all GEN_* columns

    Drop rows where SYSTEM_WIDE_GEN is NaN (future hours).
    Works for both solar and wind, and is tolerant of older schemas.
    """
    gen_region_cols = [c for c in df.columns if c.startswith("GEN_")]

    if "SYSTEM_WIDE_GEN" not in df.columns:
        if gen_region_cols:
            df["SYSTEM_WIDE_GEN"] = df[gen_region_cols].sum(axis=1, min_count=1)
            print("⚠️  'SYSTEM_WIDE_GEN' missing in file; computed as sum of GEN_* columns.")
        else:
            raise RuntimeError(
                "Expected 'SYSTEM_WIDE_GEN' or some 'GEN_*' columns, "
                "but neither were found in this dataframe."
            )

    if "DELIVERY_DATE" not in df.columns or "HOUR_ENDING" not in df.columns:
        raise RuntimeError("Expected 'DELIVERY_DATE' and 'HOUR_ENDING' columns not found.")

    keep_cols = ["DELIVERY_DATE", "HOUR_ENDING", "SYSTEM_WIDE_GEN"] + gen_region_cols

    df_gen = df[df["SYSTEM_WIDE_GEN"].notna()][keep_cols].copy()
    return df_gen

# ---------------------------------------------------
# Download + return DataFrame with robust retry
# ---------------------------------------------------

def download_report_as_df(headers: dict, href: str, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """
    Download one report with retry-on-429 logic and basic handling of transient HTTP errors.
    """
    for attempt in range(max_retries):
        resp = requests.get(href, headers=headers)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = 5 * (2 ** attempt)
            print(f"⚠️  429 Too Many Requests for {href}. Waiting {wait} seconds (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            wait = 5 * (attempt + 1)
            print(f"⚠️  HTTP error {resp.status_code} on {href}: {e}. Waiting {wait} seconds before retry...")
            time.sleep(wait)
            continue

        content = resp.content
        try:
            with ZipFile(BytesIO(content)) as z:
                member_name = z.namelist()[0]
                csv_bytes = z.read(member_name)
        except BadZipFile:
            csv_bytes = content

        return pd.read_csv(BytesIO(csv_bytes))

    raise RuntimeError(f"Failed to download {href} after {max_retries} retries.")

# ---------------------------------------------------
# Core pipeline for one report type (solar or wind)
# ---------------------------------------------------

def process_report_type(config: dict, headers: dict, now_central: datetime,
                        window_start: datetime, window_end: datetime):
    label    = config["label"]
    emil_id  = config["emil_id"]
    filename = config["filename"]

    # For archive search, go back ARCHIVE_LOOKBACK_DAYS from "now"
    start_dt = now_central - timedelta(days=ARCHIVE_LOOKBACK_DAYS)
    end_dt   = now_central

    print(f"\n=== Processing {label} ({emil_id}) ===")
    print(f"Archive search window: {start_dt} -> {end_dt}")
    archives = fetch_archives_for_window(headers, emil_id, start_dt, end_dt)
    print(f"Found {len(archives)} archives for {label}.")

    print(
        f"Selecting up to {NUM_REPORTS} reports spaced {WINDOW_HOURS} hours apart "
        f"(covering up to {TOTAL_DAYS_FROM_NOW} days back from now)..."
    )
    selected = pick_reports_every_48_hours(
        archives,
        now_dt=now_central,
        num_reports=NUM_REPORTS,
        hour_step=WINDOW_HOURS,
    )

    print(f"[{label}] Actually using {len(selected)} reports.")
    dfs = []
    for idx, (archive, post_dt) in enumerate(selected):
        href = archive.get("_links", {}).get("endpoint", {}).get("href")
        if not href:
            raise RuntimeError(f"Archive missing endpoint link: {archive}")

        print(f"[{label}] Downloading & cleaning report {idx + 1}/{len(selected)} posted at {post_dt}...")
        df_raw = download_report_as_df(headers, href)
        df_clean = clean_to_generation_only(df_raw)
        dfs.append(df_clean)

        time.sleep(SLEEP_SECONDS)

    print(f"[{label}] Combining cleaned reports into a single time series...")
    combined = pd.concat(dfs, ignore_index=True)

    combined["HOUR_ENDING"] = combined["HOUR_ENDING"].astype(int)
    combined["DELIVERY_DATE_DT"] = pd.to_datetime(combined["DELIVERY_DATE"], format="%m/%d/%Y")

    combined = combined.sort_values(
        by=["DELIVERY_DATE_DT", "HOUR_ENDING"]
    ).drop_duplicates(
        subset=["DELIVERY_DATE", "HOUR_ENDING"],
        keep="last"
    )

    # Filter to the requested window [window_start, window_end]
    start_date = window_start.date()
    end_date   = window_end.date()
    mask = (
        (combined["DELIVERY_DATE_DT"].dt.date >= start_date) &
        (combined["DELIVERY_DATE_DT"].dt.date <= end_date)
    )
    combined = combined[mask]

    combined = combined.sort_values(by=["DELIVERY_DATE_DT", "HOUR_ENDING"])
    combined = combined.drop(columns=["DELIVERY_DATE_DT"])

    # Save **directly in working directory**
    out_path = filename
    combined.to_csv(out_path, index=False)

    print(
        f"[{label}] Done. Window {start_date} -> {end_date} saved at:\n"
        f"  {os.path.abspath(out_path)}"
    )

# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("Authenticating with ERCOT Public API...")
    id_token = get_id_token(USERNAME, PASSWORD, SUBSCRIPTION_KEY)
    headers = make_headers(id_token, SUBSCRIPTION_KEY)

    now_central = datetime.now(CENTRAL_TZ)

    # Define the actual data window based on config
    window_end   = now_central - timedelta(days=WINDOW_OFFSET_DAYS)
    window_start = window_end - timedelta(days=WINDOW_LENGTH_DAYS)

    print(f"\nConfigured data window: {window_start.date()} -> {window_end.date()}")
    print(f"(length ~{WINDOW_LENGTH_DAYS} days, offset {WINDOW_OFFSET_DAYS} days back from today)")

    for config in REPORT_CONFIGS:
        process_report_type(config, headers, now_central, window_start, window_end)


if __name__ == "__main__":
    main()
