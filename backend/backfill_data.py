"""
backfill_data.py — Historical ENTSO-E Data Gap Filler
=====================================================
Batch-downloads historical Day-Ahead Prices and Actual Load from 2019–2025
to retrain the existing models, ensuring they understand post-2022 market volatility.
Script role: Historical data ingestion maintaining schema consistency.

Usage:
    python backfill_data.py --start 2019-01-01 --end 2025-12-31
    python backfill_data.py --start 2019-01-01 --end 2025-12-31 --api-key YOUR_KEY

Requires:
    pip install entsoe-py pandas
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
SPAIN_BIDDING_ZONE = "ES"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
EXISTING_DATA_FILE = PROCESSED_DIR / "spain_merged_final.csv"
OUTPUT_FILE = PROCESSED_DIR / "spain_merged_2015_2025.csv"


def fetch_monthly_data(client, year: int, month: int) -> pd.DataFrame:
    """
    Fetch one month of Day-Ahead Prices and Actual Load from ENTSO-E API.
    Returns a DataFrame with columns: [price_eur_mwh, load_mw]
    """
    start = pd.Timestamp(f"{year}-{month:02d}-01", tz="Europe/Madrid")
    if month == 12:
        end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Madrid")
    else:
        end = pd.Timestamp(f"{year}-{month + 1:02d}-01", tz="Europe/Madrid")

    logger.info(f"  Fetching {start.strftime('%Y-%m')}...")

    # Day-Ahead Prices
    try:
        prices = client.query_day_ahead_prices(SPAIN_BIDDING_ZONE, start=start, end=end)
        prices = prices.resample("1h").mean()
    except Exception as e:
        logger.warning(f"  ⚠ Prices unavailable for {year}-{month:02d}: {e}")
        prices = pd.Series(dtype=float)

    # Actual Total Load
    try:
        load = client.query_load(SPAIN_BIDDING_ZONE, start=start, end=end)
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]
        load = load.resample("1h").mean()
    except Exception as e:
        logger.warning(f"  ⚠ Load unavailable for {year}-{month:02d}: {e}")
        load = pd.Series(dtype=float)

    # Combine
    df = pd.DataFrame({
        "price_eur_mwh": prices,
        "load_mw": load
    })

    return df


def generate_synthetic_backfill(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic but realistic historical data when API is unavailable.
    Useful for testing the pipeline without an API key.
    """
    logger.info("Generating synthetic backfill data (no API key available)...")

    date_range = pd.date_range(start=start_date, end=end_date, freq="1h", tz="Europe/Madrid")
    np.random.seed(42)

    records = []
    for ts in date_range:
        h = ts.hour
        m = ts.month
        dow = ts.dayofweek

        # Price pattern (€/MWh)
        base_price = np.array([
            42, 38, 35, 33, 32, 35, 45, 65, 85, 95, 90, 82,
            78, 75, 72, 70, 72, 80, 95, 110, 105, 88, 70, 55
        ])[h]

        # Post-2022 volatility spike
        if ts.year >= 2022:
            base_price *= 1.8
        elif ts.year >= 2020:
            base_price *= 1.3

        # Seasonal
        if m in [12, 1, 2]:
            base_price *= 1.25
        elif m in [6, 7, 8]:
            base_price *= 1.15

        # Weekend
        if dow >= 5:
            base_price *= 0.82

        price = max(base_price + np.random.normal(0, 8), 0)

        # Load pattern (MW)
        base_load = np.array([
            22000, 21000, 20500, 20000, 19800, 20500,
            23000, 27000, 30000, 31500, 32000, 31000,
            30000, 29500, 29000, 28500, 29000, 30500,
            33000, 34000, 33000, 30000, 27000, 24000
        ])[h]

        if m in [7, 8]:
            base_load *= 1.1
        elif m in [12, 1, 2]:
            base_load *= 1.08
        if dow >= 5:
            base_load *= 0.88

        load = max(base_load + np.random.normal(0, 800), 15000)

        records.append({
            "timestamp": ts,
            "price_eur_mwh": round(price, 2),
            "load_mw": round(load, 0)
        })

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    return df


def backfill(start_date: str, end_date: str, api_key: str = None):
    """
    Main backfill function. Downloads historical data in monthly chunks.
    """
    logger.info("=" * 60)
    logger.info("ENTSO-E Historical Data Backfill")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 60)

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Try API first
    api_key = api_key or os.environ.get("ENTSOE_API_KEY", "")

    if api_key:
        logger.info("Using ENTSO-E API for data download...")
        try:
            from entsoe import EntsoePandasClient
            client = EntsoePandasClient(api_key=api_key)
        except ImportError:
            logger.error("entsoe-py not installed. Run: pip install entsoe-py")
            logger.info("Falling back to synthetic data generation...")
            api_key = None

    if api_key:
        # Download via API in monthly chunks
        all_data = []
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        current = start_dt
        while current <= end_dt:
            try:
                monthly = fetch_monthly_data(client, current.year, current.month)
                all_data.append(monthly)
                logger.info(f"  ✓ {current.strftime('%Y-%m')}: {len(monthly)} records")
            except Exception as e:
                logger.warning(f"  ✗ {current.strftime('%Y-%m')}: {e}")

            # Rate limiting
            time.sleep(1.5)

            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if all_data:
            new_data = pd.concat(all_data)
        


    logger.info(f"New data shape: {new_data.shape}")

    # Merge with existing data if available
    if EXISTING_DATA_FILE.exists():
        logger.info(f"Merging with existing data: {EXISTING_DATA_FILE}")
        existing = pd.read_csv(EXISTING_DATA_FILE, index_col=0, parse_dates=True)
        logger.info(f"Existing data: {existing.shape} ({existing.index.min()} to {existing.index.max()})")

        # Merge on common columns
        common_cols = ["price_eur_mwh"]
        if "load_mw" in new_data.columns and "consumption_mw" in existing.columns:
            new_data = new_data.rename(columns={"load_mw": "consumption_mw"})

        merged = pd.concat([existing, new_data], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
    else:
        merged = new_data
        logger.info("No existing data found. Using new data only.")

    # Save
    merged.to_csv(OUTPUT_FILE)
    logger.info(f"✓ Merged dataset saved to: {OUTPUT_FILE}")
    logger.info(f"  Total records: {len(merged):,}")
    logger.info(f"  Date range: {merged.index.min()} to {merged.index.max()}")
    logger.info("=" * 60)

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill ENTSO-E historical data")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--api-key", default=None, help="ENTSO-E API key (or set ENTSOE_API_KEY env var)")
    args = parser.parse_args()

    backfill(args.start, args.end, args.api_key)
