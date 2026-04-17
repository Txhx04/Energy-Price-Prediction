"""
entsoe_fetcher.py — ENTSO-E Data Acquisition (API + Simulation)
================================================================
Data ingestion from ENTSO-E API for real-time consumption/price data.
Provides two modes for fetching Spain Day-Ahead Prices and Actual Load:
  1. API Mode   — uses entsoe-py (requires ENTSOE_API_KEY env var)
  2. Simulation — generates realistic synthetic data for demo/testing

Usage:
    fetcher = ENTSOEHybridFetcher()
    prices_df = fetcher.get_day_ahead_prices("2026-04-12")
    load_df   = fetcher.get_actual_load("2026-04-12")
"""

import os
import logging
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Spain bidding zone identifier for ENTSO-E
# ─────────────────────────────────────────────────────────────
SPAIN_BIDDING_ZONE = "ES"
SPAIN_ENTSOE_CODE  = "10YES-REE------0"   # ENTSO-E area code for Spain


# ═════════════════════════════════════════════════════════════
# Abstract Base
# ═════════════════════════════════════════════════════════════
class ENTSOEFetcherBase(ABC):
    """Abstract interface for ENTSO-E data fetchers."""

    @abstractmethod
    def get_day_ahead_prices(self, date_str: str) -> pd.DataFrame:
        """
        Fetch 24-hour Day-Ahead prices for Spain.
        Returns DataFrame with columns: [hour, price_eur_mwh]
        """
        pass

    @abstractmethod
    def get_actual_load(self, date_str: str) -> pd.DataFrame:
        """
        Fetch 24-hour Actual Load for Spain.
        Inputs: date_str (YYYY-MM-DD string)
        Outputs: DataFrame with schema columns: [hour, load_mw]
        ML Principle: Data acquisition mapping raw API payloads constraints (JSON/XML) 
        into stationary continuous numerical features for forecasting. Annotates temporal gap handling.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this fetcher is currently usable."""
        pass


# ═════════════════════════════════════════════════════════════
# 1. API Mode — entsoe-py
# ═════════════════════════════════════════════════════════════
class ENTSOEApiFetcher(ENTSOEFetcherBase):
    """
    Fetches data from ENTSO-E Transparency Platform REST API.
    Requires the ENTSOE_API_KEY environment variable.
    """

    def __init__(self):
        self.api_key = os.environ.get("ENTSOE_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from entsoe import EntsoePandasClient
                self._client = EntsoePandasClient(api_key=self.api_key)
            except ImportError:
                logger.warning("entsoe-py not installed. Install with: pip install entsoe-py")
                raise
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_day_ahead_prices(self, date_str: str) -> pd.DataFrame:
        """Fetch Day-Ahead prices via ENTSO-E API."""
        client = self._get_client()
        start = pd.Timestamp(date_str, tz="Europe/Madrid")
        end   = start + pd.Timedelta(days=1)

        logger.info(f"[API] Fetching Day-Ahead prices for {date_str}")
        raw = client.query_day_ahead_prices(SPAIN_BIDDING_ZONE, start=start, end=end)

        df = pd.DataFrame({
            "hour": list(range(24)),
            "price_eur_mwh": raw.values[:24]
        })
        return df

    def get_actual_load(self, date_str: str) -> pd.DataFrame:
        """Fetch Actual Total Load via ENTSO-E API."""
        client = self._get_client()
        start = pd.Timestamp(date_str, tz="Europe/Madrid")
        end   = start + pd.Timedelta(days=1)

        logger.info(f"[API] Fetching Actual Load for {date_str}")
        raw = client.query_load(SPAIN_BIDDING_ZONE, start=start, end=end)

        # Resample to hourly if sub-hourly
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        hourly = raw.resample("1h").mean()

        df = pd.DataFrame({
            "hour": list(range(24)),
            "load_mw": hourly.values[:24]
        })
        return df

    def get_load_forecast(self, date_str: str) -> pd.DataFrame:
        """Fetch Day-Ahead Load Forecast via ENTSO-E API."""
        client = self._get_client()
        start = pd.Timestamp(date_str, tz="Europe/Madrid")
        end   = start + pd.Timedelta(days=1)

        logger.info(f"[API] Fetching Load Forecast for {date_str}")
        raw = client.query_load_forecast(SPAIN_BIDDING_ZONE, start=start, end=end)

        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        hourly = raw.resample("1h").mean()

        df = pd.DataFrame({
            "hour": list(range(24)),
            "load_forecast_mw": hourly.values[:24]
        })
        return df

    def get_generation_forecast(self, date_str: str) -> pd.DataFrame:
        """Fetch Day-Ahead Wind & Solar Forecast via ENTSO-E API."""
        client = self._get_client()
        start = pd.Timestamp(date_str, tz="Europe/Madrid")
        end   = start + pd.Timedelta(days=1)

        logger.info(f"[API] Fetching Wind/Solar Forecast for {date_str}")
        raw = client.query_wind_and_solar_forecast(SPAIN_BIDDING_ZONE, start=start, end=end, process_type='A01')

        # raw is typically a DataFrame with 'Wind Onshore' and 'Solar'
        hourly = raw.resample("1h").mean()

        wind_col = [c for c in hourly.columns if "Wind Onshore" in c]
        solar_col = [c for c in hourly.columns if "Solar" in c]
        
        wind_vals = hourly[wind_col[0]].values[:24] if wind_col else np.zeros(24)
        solar_vals = hourly[solar_col[0]].values[:24] if solar_col else np.zeros(24)

        df = pd.DataFrame({
            "hour": list(range(24)),
            "forecast_wind_onshore_mw": wind_vals,
            "forecast_solar_mw": solar_vals
        })
        return df


# ═════════════════════════════════════════════════════════════
# 2. Simulation Mode — Realistic Synthetic Data
# ═════════════════════════════════════════════════════════════
class ENTSOESimulatedFetcher(ENTSOEFetcherBase):
    """
    Generates realistic synthetic Spain energy data for demo/testing.
    Based on typical Spanish electricity price and consumption patterns.
    """

    def is_available(self) -> bool:
        return True   # Always available as the ultimate fallback

    def get_day_ahead_prices(self, date_str: str) -> pd.DataFrame:
        """Generate realistic Day-Ahead price curve for Spain."""
        np.random.seed(hash(date_str) % (2**31))
        dt = datetime.strptime(date_str, "%Y-%m-%d")

        # Typical Spanish Day-Ahead pattern (€/MWh)
        # Off-peak ~40-60, morning peak ~80-120, evening peak ~90-140
        base_profile = np.array([
            42, 38, 35, 33, 32, 35,   # 00:00 - 05:00 (night valley)
            45, 65, 85, 95, 90, 82,   # 06:00 - 11:00 (morning ramp)
            78, 75, 72, 70, 72, 80,   # 12:00 - 17:00 (afternoon)
            95, 110, 105, 88, 70, 55  # 18:00 - 23:00 (evening peak)
        ], dtype=float)

        # Seasonal adjustment
        month = dt.month
        if month in [12, 1, 2]:      # Winter: higher prices
            base_profile *= 1.25
        elif month in [6, 7, 8]:     # Summer: slightly higher (AC demand)
            base_profile *= 1.15
        elif month in [3, 4, 5, 9, 10, 11]:  # Shoulder: baseline
            base_profile *= 1.0

        # Weekend discount
        if dt.weekday() >= 5:
            base_profile *= 0.82

        # Add noise
        noise = np.random.normal(0, 5, 24)
        prices = np.maximum(base_profile + noise, 5)  # Floor at 5 €/MWh

        return pd.DataFrame({
            "hour": list(range(24)),
            "price_eur_mwh": np.round(prices, 2)
        })

    def get_actual_load(self, date_str: str) -> pd.DataFrame:
        """Generate realistic Actual Load curve for Spain (MW)."""
        np.random.seed((hash(date_str) + 42) % (2**31))
        dt = datetime.strptime(date_str, "%Y-%m-%d")

        # Typical Spain national load profile (MW)
        base_profile = np.array([
            22000, 21000, 20500, 20000, 19800, 20500,  # 00-05
            23000, 27000, 30000, 31500, 32000, 31000,  # 06-11
            30000, 29500, 29000, 28500, 29000, 30500,  # 12-17
            33000, 34000, 33000, 30000, 27000, 24000   # 18-23
        ], dtype=float)

        # Seasonal adjustment
        month = dt.month
        if month in [7, 8]:
            base_profile *= 1.1  # Summer AC
        elif month in [12, 1, 2]:
            base_profile *= 1.08  # Winter heating

        # Weekend reduction
        if dt.weekday() >= 5:
            base_profile *= 0.88

        # Add noise
        noise = np.random.normal(0, 500, 24)
        load = np.maximum(base_profile + noise, 15000)

        return pd.DataFrame({
            "hour": list(range(24)),
            "load_mw": np.round(load, 0)
        })

    def get_load_forecast(self, date_str: str) -> pd.DataFrame:
        """Generate simulated load forecast based on actual load simulation."""
        actual_df = self.get_actual_load(date_str)
        # Add 3% error to make a realistic forecast
        np.random.seed(hash(date_str) % (2**31) + 123)
        noise = np.random.normal(0, actual_df["load_mw"].mean() * 0.03, 24)
        forecast_load = np.maximum(actual_df["load_mw"] + noise, 15000)
        
        return pd.DataFrame({
            "hour": list(range(24)),
            "load_forecast_mw": forecast_load
        })
        
    def get_generation_forecast(self, date_str: str) -> pd.DataFrame:
        """Generate simulated solar and wind forecasts."""
        np.random.seed(hash(date_str) % (2**31) + 456)
        
        # Solar peaks at noon
        solar_base = np.array([
            0, 0, 0, 0, 0, 0, 
            200, 1500, 4000, 7000, 9000, 10000,
            10500, 10000, 8500, 6000, 3500, 1000,
            100, 0, 0, 0, 0, 0
        ], dtype=float)
        
        noise_solar = np.random.normal(0, 500, 24)
        solar = np.maximum(solar_base + noise_solar, 0)
        solar[solar_base == 0] = 0  # Force 0 at night
        
        # Wind is more random but often higher at night
        wind_base = np.random.normal(8000, 3000, 24)
        wind_base += np.sin(np.linspace(0, 2*np.pi, 24)) * 2000
        wind = np.maximum(wind_base, 1000)

        return pd.DataFrame({
            "hour": list(range(24)),
            "forecast_wind_onshore_mw": wind,
            "forecast_solar_mw": solar
        })

# ═════════════════════════════════════════════════════════════
# 3. Hybrid Fetcher — Orchestrator (API → Simulation)
# ═════════════════════════════════════════════════════════════
class ENTSOEHybridFetcher:
    """
    Tries fetchers in order: API → Simulation.
    Uses ENTSO-E API when ENTSOE_API_KEY is set in .env,
    falls back to realistic simulation for demo/testing.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._api_fetcher = ENTSOEApiFetcher()
        self._sim_fetcher = ENTSOESimulatedFetcher()

        # Build priority chain: API first, then simulation fallback
        self._fetchers = []
        if self._api_fetcher.is_available():
            self._fetchers.append(("API", self._api_fetcher))
            logger.info("✓ ENTSO-E API key detected — API mode enabled")
        else:
            logger.info("⚠ No ENTSO-E API key found — set ENTSOE_API_KEY in .env")
        # Simulation is always last
        self._fetchers.append(("Simulation", self._sim_fetcher))
        logger.info("✓ Simulation fallback always available")

    def _fetch_with_fallback(self, method_name: str, date_str: str) -> pd.DataFrame:
        """Try each fetcher in priority order with retries."""
        last_error = None
        for name, fetcher in self._fetchers:
            for attempt in range(1, self.max_retries + 1):
                try:
                    method = getattr(fetcher, method_name)
                    result = method(date_str)
                    logger.info(f"[{name}] Successfully fetched {method_name} for {date_str}")
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[{name}] {method_name} attempt {attempt}/{self.max_retries} "
                        f"failed: {e}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(1)  # Short backoff

        raise RuntimeError(
            f"All fetchers failed for {method_name}({date_str}). "
            f"Last error: {last_error}"
        )

    def get_day_ahead_prices(self, date_str: str) -> pd.DataFrame:
        return self._fetch_with_fallback("get_day_ahead_prices", date_str)

    def get_actual_load(self, date_str: str) -> pd.DataFrame:
        return self._fetch_with_fallback("get_actual_load", date_str)

    def get_load_forecast(self, date_str: str) -> pd.DataFrame:
        return self._fetch_with_fallback("get_load_forecast", date_str)

    def get_generation_forecast(self, date_str: str) -> pd.DataFrame:
        return self._fetch_with_fallback("get_generation_forecast", date_str)

    def get_active_mode(self) -> str:
        """Return which mode will be tried first."""
        return self._fetchers[0][0] if self._fetchers else "None"
