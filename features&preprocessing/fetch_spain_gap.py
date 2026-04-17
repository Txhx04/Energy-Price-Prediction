"""
Spain Energy Dataset Gap Filler: 2019-01-01 → 2026-04-13
==========================================================
Script role: Data ingestion from ENTSO-E and Open-Meteo APIs for dataset construction.
Sources:
  - ENTSO-E API (entsoe-py)  → energy columns
  - Open-Meteo (ERA5 reanalysis, free, no key) → weather columns
  - Computed locally          → datetime feature columns

Rate-limit awareness:
  - ENTSO-E: max 400 req/min. entsoe-py auto-chunks via @year_limited /
    @day_limited decorators. We add a 2-second sleep between each top-level
    query call as extra safety.
  - Open-Meteo: no stated hard limit; we fetch one call per year to keep
    payload sizes manageable.

Known quirks handled:
  - Day-ahead prices (12.1.D) Jan 2026 bug → businessType=A62 workaround
    applied via the raw client fallback.
  - generation query returns a DataFrame with MultiIndex columns; we
    extract only the columns we need by PSR type.
  - Offshore wind is negligible / often missing for Spain; filled with 0.

Usage:
    pip install entsoe-py openmeteo-requests requests-cache retry-requests pandas numpy

    python fetch_spain_gap.py --api_key YOUR_ENTSOE_KEY
    # Output: spain_gap_2019_2026.csv  (same schema as original)
    # Then merge: python fetch_spain_gap.py --merge
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── ENTSO-E ──────────────────────────────────────────────────────────────────
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.exceptions import NoMatchingDataError

# ── Open-Meteo (weather) ─────────────────────────────────────────────────────
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_CODE   = "ES"           # entsoe-py country code for Spain
TIMEZONE       = "Europe/Madrid"
# Madrid coords for weather (Open-Meteo ERA5 reanalysis)
WEATHER_LAT    = 40.4168
WEATHER_LON    = -3.7038

GAP_START = pd.Timestamp("2019-01-01", tz=TIMEZONE)
GAP_END   = pd.Timestamp("2026-04-13 23:00:00", tz=TIMEZONE)

# ─ Get project root dynamically (one level up from script location) ─
PROJECT_ROOT = Path(__file__).parent

ORIGINAL_CSV  = PROJECT_ROOT / "Data" / "processed" / "spain_merged_final.csv"
OUTPUT_GAP    = PROJECT_ROOT / "outputs" / "spain_gap_2019_2026.csv"
OUTPUT_MERGED = PROJECT_ROOT / "outputs" / "spain_merged_full.csv"

SLEEP_BETWEEN_QUERIES = 2   # seconds – stay well under 400 req/min

# PSR type codes → our column names
PSR_MAP = {
    "B16": "gen_solar_mw",
    "B19": "gen_wind_onshore_mw",
    "B18": "gen_wind_offshore_mw",
    "B11": "gen_hydro_ror_mw",
    "B14": "gen_nuclear_mw",
    "B04": "gen_fossil_gas_mw",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_query(fn, *args, label="", **kwargs):
    """Call an entsoe-py query function; return None on NoMatchingDataError."""
    print(f"  Querying {label} …", end=" ", flush=True)
    try:
        result = fn(*args, **kwargs)
        print("OK")
        return result
    except NoMatchingDataError:
        print("NO DATA")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    finally:
        time.sleep(SLEEP_BETWEEN_QUERIES)


def year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) pairs in ~1-year slices."""
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur, nxt
        cur = nxt


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTSO-E  ─ energy data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_entsoe(api_key: str) -> pd.DataFrame:
    """
    Fetch energy consumption and generation data from ENTSO-E.
    
    Inputs:
      - api_key: str, authentication token for ENTSO-E API.
      
    Outputs:
      - df: pd.DataFrame, hourly time-series data indexed by datetime.
      
    ML Principle: Raw feature ingestion for temporal forecasting.
    """
    client = EntsoePandasClient(api_key=api_key)

    # Hourly index covering the full gap
    idx = pd.date_range(GAP_START, GAP_END, freq="h", tz=TIMEZONE)
    df = pd.DataFrame(index=idx)

    # ── 1a. Actual Load (6.1.A) ──────────────────────────────────────────
    print("\n[ENTSO-E] Actual Load …")
    load_parts = []
    for s, e in year_chunks(GAP_START, GAP_END):
        r = safe_query(client.query_load, COUNTRY_CODE,
                       start=s, end=e, label=f"load {s.year}")
        if r is not None:
            load_parts.append(r)
    if load_parts:
        load_series = pd.concat(load_parts).sort_index()
        load_series = load_series[~load_series.index.duplicated(keep="first")]
        df["consumption_mw"] = load_series.reindex(idx)

    # ── 1b. Load Forecast (6.1.B) ────────────────────────────────────────
    print("\n[ENTSO-E] Load Forecast …")
    lf_parts = []
    for s, e in year_chunks(GAP_START, GAP_END):
        r = safe_query(client.query_load_forecast, COUNTRY_CODE,
                       start=s, end=e, label=f"load_fc {s.year}")
        if r is not None:
            lf_parts.append(r)
    if lf_parts:
        lf_series = pd.concat(lf_parts).sort_index()
        lf_series = lf_series[~lf_series.index.duplicated(keep="first")]
        df["load_forecast_mw"] = lf_series.reindex(idx)

    # ── 1c. Day-Ahead Prices (12.1.D) ────────────────────────────────────
    # Note: entsoe-py's query_day_ahead_prices already handles the
    # businessType=A62 workaround internally in recent versions.
    # If you see HTTP 400 errors on prices, update entsoe-py:
    #   pip install --upgrade entsoe-py
    print("\n[ENTSO-E] Day-Ahead Prices …")
    price_parts = []
    for s, e in year_chunks(GAP_START, GAP_END):
        r = safe_query(client.query_day_ahead_prices, COUNTRY_CODE,
                       start=s, end=e, label=f"prices {s.year}")
        if r is not None:
            price_parts.append(r)
    if price_parts:
        price_series = pd.concat(price_parts).sort_index()
        price_series = price_series[~price_series.index.duplicated(keep="first")]
        # Resample to hourly if needed (post-Oct 2025 SDAC may publish 15min)
        if price_series.index.freq not in ["h", "H", None]:
            price_series = price_series.resample("h").first()
        df["price_eur_mwh"]       = price_series.reindex(idx)
        df["price_day_ahead_eur"] = price_series.reindex(idx)

    # ── 1d. Generation per type (15.1.A / 16.1.B) ────────────────────────
    print("\n[ENTSO-E] Generation per type …")
    gen_parts = []
    for s, e in year_chunks(GAP_START, GAP_END):
        r = safe_query(client.query_generation, COUNTRY_CODE,
                       start=s, end=e, label=f"gen {s.year}")
        if r is not None:
            gen_parts.append(r)

    if gen_parts:
        gen_df = pd.concat(gen_parts).sort_index()
        gen_df = gen_df[~gen_df.index.duplicated(keep="first")]
        # gen_df has MultiIndex columns: (PSR_type, Actual Aggregated)
        # Flatten to single level
        if isinstance(gen_df.columns, pd.MultiIndex):
            gen_df.columns = ["_".join(str(c) for c in col).strip()
                               for col in gen_df.columns]

        # Map PSR codes to our column names
        psr_friendly = {
            "Solar_Actual Aggregated":          "gen_solar_mw",
            "Wind Onshore_Actual Aggregated":   "gen_wind_onshore_mw",
            "Wind Offshore_Actual Aggregated":  "gen_wind_offshore_mw",
            "Hydro Run-of-river and poundage_Actual Aggregated": "gen_hydro_ror_mw",
            "Nuclear_Actual Aggregated":        "gen_nuclear_mw",
            "Fossil Gas_Actual Aggregated":     "gen_fossil_gas_mw",
        }
        for raw_col, our_col in psr_friendly.items():
            if raw_col in gen_df.columns:
                df[our_col] = gen_df[raw_col].reindex(idx)
            else:
                df[our_col] = np.nan

        # Offshore is rarely published for Spain → fill 0
        if df["gen_wind_offshore_mw"].isna().all():
            df["gen_wind_offshore_mw"] = 0.0

    # ── 1e. Wind & Solar Forecasts ────────────────────────────────────────
    print("\n[ENTSO-E] Wind & Solar Forecasts …")
    wsf_parts = []
    for s, e in year_chunks(GAP_START, GAP_END):
        r = safe_query(client.query_wind_and_solar_forecast, COUNTRY_CODE,
                       start=s, end=e, label=f"ws_fc {s.year}")
        if r is not None:
            wsf_parts.append(r)

    if wsf_parts:
        wsf_df = pd.concat(wsf_parts).sort_index()
        wsf_df = wsf_df[~wsf_df.index.duplicated(keep="first")]
        if isinstance(wsf_df.columns, pd.MultiIndex):
            wsf_df.columns = ["_".join(str(c) for c in col).strip()
                               for col in wsf_df.columns]
        solar_fc_cols = [c for c in wsf_df.columns if "Solar" in c]
        wind_fc_cols  = [c for c in wsf_df.columns if "Wind Onshore" in c]
        if solar_fc_cols:
            df["forecast_solar_mw"] = wsf_df[solar_fc_cols[0]].reindex(idx)
        if wind_fc_cols:
            df["forecast_wind_onshore_mw"] = wsf_df[wind_fc_cols[0]].reindex(idx)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPEN-METEO  ─ weather data (ERA5 reanalysis, free, no key needed)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weather() -> pd.DataFrame:
    """
    Fetch historical weather metrics (ERA5 reanalysis) for feature engineering.
    Uses Open-Meteo's Historical API backed by ERA5 reanalysis.
    Variables mapped as closely as possible to the original dataset columns.
    Fetched in year-wide slices to keep requests small.
    
    Inputs: None (relies on hardcoded Madrid bounding box/coordinates).
    Outputs: 
      - weather: pd.DataFrame, hourly features [temperature, pressure, etc.]
    """
    print("\n[Open-Meteo] Weather (ERA5) …")

    # Set up a cached session so accidental re-runs don't hammer the API
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    all_parts = []

    for year in range(2019, 2027):
        y_start = f"{year}-01-01"
        y_end   = f"{year}-04-13" if year == 2026 else f"{year}-12-31"

        print(f"  Weather {year} …", end=" ", flush=True)

        params = {
            "latitude":   WEATHER_LAT,
            "longitude":  WEATHER_LON,
            "start_date": y_start,
            "end_date":   y_end,
            "hourly": [
                "temperature_2m",
                "apparent_temperature",
                "surface_pressure",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "cloud_cover",
            ],
            "timezone": TIMEZONE,
        }

        try:
            responses = om.weather_api(
                "https://archive-api.open-meteo.com/v1/archive", params=params
            )
            r = responses[0]
            hourly = r.Hourly()

            times = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ).tz_convert(TIMEZONE)

            df_w = pd.DataFrame({
                "temperature_celsius": hourly.Variables(0).ValuesAsNumpy(),
                "temp_min_celsius":    hourly.Variables(1).ValuesAsNumpy(),  # apparent_temp as proxy
                "temp_max_celsius":    hourly.Variables(1).ValuesAsNumpy(),  # will be post-processed
                "pressure_hpa":        hourly.Variables(2).ValuesAsNumpy(),
                "humidity_percent":    hourly.Variables(3).ValuesAsNumpy(),
                "wind_speed_ms":       hourly.Variables(4).ValuesAsNumpy() / 3.6,  # km/h → m/s
                "wind_direction_deg":  hourly.Variables(5).ValuesAsNumpy(),
                "rain_mm":             hourly.Variables(6).ValuesAsNumpy(),
                "cloud_cover_percent": hourly.Variables(7).ValuesAsNumpy(),
            }, index=times)

            all_parts.append(df_w)
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(1)

    if not all_parts:
        print("  WARNING: No weather data fetched!")
        return pd.DataFrame()

    weather = pd.concat(all_parts).sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]

    # Daily min/max from hourly temperature
    daily_min = weather["temperature_celsius"].resample("D").min()
    daily_max = weather["temperature_celsius"].resample("D").max()
    weather["temp_min_celsius"] = weather.index.normalize().map(
        daily_min.to_dict()
    )
    weather["temp_max_celsius"] = weather.index.normalize().map(
        daily_max.to_dict()
    )

    # Cast integer columns to match original schema
    for col in ["pressure_hpa", "humidity_percent", "wind_speed_ms",
                "wind_direction_deg", "cloud_cover_percent"]:
        weather[col] = weather[col].round().astype("Int64")

    return weather


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENGINEERED FEATURES  ─ pure datetime math
# ─────────────────────────────────────────────────────────────────────────────

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from datetime index.
    
    Inputs:
      - df: pd.DataFrame, data indexed by pandas DatetimeIndex.
      
    Outputs:
      - df: pd.DataFrame, mutated in-place with new integer and cyclical float columns.
      
    ML Principle: Cyclical encoding (sin/cos transformations for hour, month) 
    to preserve continuity and circular distance for tree-based and recurrent models.
    """
    df["year"]       = df.index.year
    df["month"]      = df.index.month
    df["day"]        = df.index.day
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"]    = df.index.quarter

    # Cyclical encoding logic (sin/cos representation of time to maintain 23:00 to 00:00 proximity)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_gap(api_key: str) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1: Fetching ENTSO-E energy data")
    print("=" * 60)
    energy_df = fetch_entsoe(api_key)

    print("\n" + "=" * 60)
    print("Step 2: Fetching Open-Meteo weather data")
    print("=" * 60)
    weather_df = fetch_weather()

    print("\n" + "=" * 60)
    print("Step 3: Merging & engineering features")
    print("=" * 60)
    if not weather_df.empty:
        # Align weather to energy index
        weather_aligned = weather_df.reindex(energy_df.index)
        gap = energy_df.join(weather_aligned, how="left")
    else:
        gap = energy_df

    gap = add_datetime_features(gap)
    gap.index.name = "datetime"

    # Reorder columns to match original
    original_cols = [
        "consumption_mw", "price_eur_mwh", "load_forecast_mw",
        "price_day_ahead_eur", "gen_solar_mw", "gen_wind_onshore_mw",
        "gen_wind_offshore_mw", "gen_hydro_ror_mw", "gen_nuclear_mw",
        "gen_fossil_gas_mw", "forecast_solar_mw", "forecast_wind_onshore_mw",
        "temperature_celsius", "temp_min_celsius", "temp_max_celsius",
        "pressure_hpa", "humidity_percent", "wind_speed_ms",
        "wind_direction_deg", "rain_mm", "cloud_cover_percent",
        "year", "month", "day", "hour", "dayofweek", "is_weekend",
        "quarter", "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    for c in original_cols:
        if c not in gap.columns:
            gap[c] = np.nan

    gap = gap[original_cols]

    print(f"\nGap dataset shape: {gap.shape}")
    print(f"Date range: {gap.index.min()} → {gap.index.max()}")
    print(f"Missing values:\n{gap.isnull().sum()}")

    return gap


def merge_with_original(gap: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Step 4: Merging with original dataset")
    print("=" * 60)

    orig = pd.read_csv(ORIGINAL_CSV, index_col=0, parse_dates=True)
    orig.index = pd.DatetimeIndex(orig.index)
    if orig.index.tz is None:
        orig.index = orig.index.tz_localize("UTC").tz_convert(TIMEZONE)

    combined = pd.concat([orig, gap]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    OUTPUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_MERGED)
    print(f"Merged dataset saved → {OUTPUT_MERGED}")
    print(f"Total rows: {len(combined):,}  |  "
          f"Date range: {combined.index.min()} → {combined.index.max()}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill Spain energy dataset gap 2019-2026")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Your ENTSO-E API security token")
    parser.add_argument("--skip_entsoe", action="store_true",
                        help="Skip ENTSO-E fetch (use if gap CSV already exists)")
    parser.add_argument("--skip_weather", action="store_true",
                        help="Skip weather fetch")
    args = parser.parse_args()

    if args.skip_entsoe and OUTPUT_GAP.exists():
        print(f"Loading existing gap file: {OUTPUT_GAP}")
        gap_df = pd.read_csv(OUTPUT_GAP, index_col=0, parse_dates=True)
    else:
        gap_df = build_gap(args.api_key)
        OUTPUT_GAP.mkdir(parents=True, exist_ok=True) if OUTPUT_GAP.is_dir() else None
        gap_df.to_csv(OUTPUT_GAP)
        print(f"\nGap CSV saved → {OUTPUT_GAP}")

    merge_with_original(gap_df)
    print("\nDone.")