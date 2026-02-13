"""
Data Ingestion Module
Sensex CNN-LSTM Forecasting Engine
"""

import os
import time
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SensexDataLoader:
    """
    Handles ingestion of Sensex data from Yahoo Finance.
    """

    def __init__(self, ticker: str = "^BSESN", years: int = 12):
        self.ticker = ticker
        self.years = years

        self.end_date = datetime.today()
        self.start_date = datetime(
            self.end_date.year - years,
            self.end_date.month,
            self.end_date.day
        )

        self.raw_data_path = "data/raw/sensex_raw.csv"

    def fetch_data(self) -> pd.DataFrame:
        """
        Robust Yahoo Finance fetch with retries and fallback.
        """

        logging.info(f"Fetching data for {self.ticker}")

        df = None

        for attempt in range(3):
            try:
                df = yf.download(
                    self.ticker,
                    start=self.start_date.strftime("%Y-%m-%d"),
                    end=self.end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=False,
                    progress=False
                )

                if df is not None and not df.empty:
                    break

            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)

        # Fallback
        if df is None or df.empty:
            logging.warning("Trying fallback method using period...")
            df = yf.download(
                self.ticker,
                period="12y",
                interval="1d",
                auto_adjust=False,
                progress=False
            )

        if df is None or df.empty:
            raise ValueError(
                f"Yahoo Finance returned empty data for {self.ticker}."
            )

        df.reset_index(inplace=True)

        # Flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        logging.info(f"Fetched {len(df)} rows.")

        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate required columns.
        """

        required_columns = [
            "Date", "Open", "High", "Low",
            "Close", "Adj Close", "Volume"
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        df.dropna(inplace=True)

        logging.info("Data validation successful.")

        return df

    def save_raw_data(self, df: pd.DataFrame) -> None:
        """
        Save raw CSV.
        """

        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        df.to_csv(self.raw_data_path, index=False)

        logging.info(f"Raw data saved to {self.raw_data_path}")

    def run(self) -> pd.DataFrame:
        """
        Complete ingestion pipeline.
        """

        df = self.fetch_data()
        df = self.validate_data(df)
        self.save_raw_data(df)

        return df
