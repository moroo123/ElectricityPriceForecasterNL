import pandas as pd
from entsoe_client import EntsoeDataClient
from cache import DataCache


def download_all_data(start_year: int = 2022, end_year: int = 2024, bidding_zone: str = "NL"):
    client = EntsoeDataClient(bidding_zone=bidding_zone)
    cache = DataCache()

    data_types = {
        "prices": client.get_day_ahead_prices,
        "load_forecast": client.get_load_forecast,
        "actual_load": client.get_actual_load,
        "wind_onshore": client.get_wind_forecast,
        "wind_offshore": client.get_wind_offshore_forecast,
        "solar": client.get_solar_forecast,
    }

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Calculate month boundaries
            start = pd.Timestamp(year=year, month=month,
                                 day=1, tz="Europe/Amsterdam")
            if month == 12:
                end = pd.Timestamp(year=year + 1, month=1,
                                   day=1, tz="Europe/Amsterdam")
            else:
                end = pd.Timestamp(year=year, month=month + 1,
                                   day=1, tz="Europe/Amsterdam")

            print(f"Fetching {start.strftime('%Y-%m')}...")

            for data_type, fetch_func in data_types.items():
                # Check what's already cached
                result = cache.get_data(data_type, start, end, bidding_zone)

                if result["coverage"] == 1.0:
                    print(f"  {data_type}: already cached")
                    continue

                try:
                    data = fetch_func(start, end)
                    cache.store_data(data_type, data, bidding_zone)
                    print(f"  {data_type}: downloaded and cached")
                except Exception as e:
                    print(f"  {data_type}: ERROR - {e}")

            print()


if __name__ == "__main__":
    download_all_data()
