from entsoe import EntsoePandasClient
import pandas as pd
from dotenv import load_dotenv
import os


class EntsoeDataClient:
    def __init__(self, bidding_zone: str = "NL"):
        load_dotenv()
        self.client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
        self.bidding_zone = bidding_zone

    def get_day_ahead_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch day-ahead prices for the bidding zone."""
        result = self.client.query_day_ahead_prices(
            self.bidding_zone,
            start=start,
            end=end
        )
        return self._to_series(result, name="prices")

    def get_load_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch TSO load forecast."""
        result = self.client.query_load_forecast(
            self.bidding_zone,
            start=start,
            end=end
        )
        return self._to_series(result, name="load_forecast")

    def get_actual_load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch actual measured load."""
        result = self.client.query_load(
            self.bidding_zone,
            start=start,
            end=end
        )
        return self._to_series(result, name="actual_load")

    def get_wind_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch day-ahead wind generation forecast."""
        result = self.client.query_wind_and_solar_forecast(
            self.bidding_zone,
            start=start,
            end=end,
            psr_type="B19"  # Wind Onshore
        )
        return self._to_series(result, name="wind_onshore")

    def get_solar_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch day-ahead solar generation forecast."""
        result = self.client.query_wind_and_solar_forecast(
            self.bidding_zone,
            start=start,
            end=end,
            psr_type="B16"  # Solar
        )
        return self._to_series(result, name="solar")

    def get_wind_offshore_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Fetch day-ahead offshore wind forecast (relevant for NL)."""
        result = self.client.query_wind_and_solar_forecast(
            self.bidding_zone,
            start=start,
            end=end,
            psr_type="B18"  # Wind Offshore
        )
        return self._to_series(result, name="wind_offshore")

    def _to_series(self, result, name: str) -> pd.Series:
        """Convert API result to Series, handling both Series and DataFrame returns."""
        if isinstance(result, pd.DataFrame):
            if result.shape[1] == 1:
                return result.squeeze().rename(name)
            else:
                return result.sum(axis=1).rename(name)
        return result.rename(name)
