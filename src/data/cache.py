import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import Config


class DataCache:
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Config.DB_PATH
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            for table in ["prices", "load_forecast", "actual_load",
                          "wind_onshore", "wind_offshore", "solar"]:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        timestamp TEXT PRIMARY KEY,
                        value REAL,
                        bidding_zone TEXT
                    )
                """)

    def get_data(self, data_type: str, start: pd.Timestamp, end: pd.Timestamp,
                 bidding_zone: str = "NL") -> dict:
        """     
        Generic retrieval for any data type.
        """
        valid_types = ["prices", "load_forecast", "actual_load",
                       "wind_onshore", "wind_offshore", "solar"]
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")

        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT timestamp, value 
                FROM {data_type} 
                WHERE bidding_zone = ?
                AND timestamp >= ?
                AND timestamp < ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(bidding_zone, start.isoformat(), end.isoformat())
            )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            df.index = df.index.tz_convert(start.tz)

        # Detect frequency from cached data, or assume hourly
        if not df.empty and len(df) > 1:
            # Infer frequency from actual data
            time_diffs = df.index.to_series().diff().dropna()
            most_common_diff = time_diffs.mode().iloc[0]
            freq = most_common_diff
        else:
            freq = pd.Timedelta(hours=1)

        # Build expected index based on detected frequency
        expected_index = pd.date_range(
            start=start, end=end, freq=freq, inclusive="left")

        cached_timestamps = set(df.index) if not df.empty else set()
        missing_timestamps = sorted(set(expected_index) - cached_timestamps)
        missing_ranges = self._timestamps_to_ranges(missing_timestamps)
        coverage = len(cached_timestamps) / \
            len(expected_index) if len(expected_index) > 0 else 1.0

        return {
            "data": df,
            "missing_ranges": missing_ranges,
            "coverage": coverage
        }

    def store_data(self, data_type: str, data: pd.Series, bidding_zone: str = "NL"):
        """Generic storage for any data type."""
        valid_types = ["prices", "load_forecast", "actual_load",
                       "wind_onshore", "wind_offshore", "solar"]
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")

        if data.empty:
            return

        records = []
        for ts, value in data.items():
            # Handle both Timestamp and string indices
            if hasattr(ts, 'isoformat'):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            records.append((ts_str, value, bidding_zone))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"""
                INSERT INTO {data_type} (timestamp, value, bidding_zone)
                VALUES (?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET value = excluded.value
                """,
                records
            )
            conn.commit()

    def _timestamps_to_ranges(self, timestamps: list) -> list:
        """Convert list of timestamps to list of (start, end) contiguous ranges."""
        if not timestamps:
            return []

        ranges = []
        range_start = timestamps[0]
        prev = timestamps[0]

        for ts in timestamps[1:]:
            # If gap is more than 1 hour, start new range
            if ts - prev > pd.Timedelta(hours=1):
                ranges.append((range_start, prev + pd.Timedelta(hours=1)))
                range_start = ts
            prev = ts

        # Close final range
        ranges.append((range_start, prev + pd.Timedelta(hours=1)))

        return ranges

    def get_last_timestamp(self, data_type: str, bidding_zone: str = "NL") -> pd.Timestamp:
        """
        Get the last (maximum) timestamp for a given data type.
        
        Parameters
        ----------
        data_type : str
            Type of data (prices, load_forecast, etc.)
        bidding_zone : str, default="NL"
            Bidding zone
            
        Returns
        -------
        pd.Timestamp or None
            The last timestamp in UTC, or None if table is empty
        """
        valid_types = ["prices", "load_forecast", "actual_load",
                       "wind_onshore", "wind_offshore", "solar"]
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")
            
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT MAX(timestamp) FROM {data_type} WHERE bidding_zone = ?"
            cursor = conn.cursor()
            cursor.execute(query, (bidding_zone,))
            result = cursor.fetchone()
            
        if result and result[0]:
            return pd.Timestamp(result[0]).tz_convert("UTC")
        return None
