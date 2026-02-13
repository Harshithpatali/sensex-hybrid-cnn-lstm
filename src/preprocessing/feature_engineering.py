"""
Feature Engineering Module
Sensex CNN-LSTM Forecasting Engine
"""

import os
import logging
import numpy as np
import pandas as pd
import ta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineer:
    """
    Handles feature engineering and regime labeling.
    """

    def __init__(self):
        self.processed_path = "data/processed/sensex_features.csv"

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators and engineered features.
        """

        logging.info("Starting feature engineering...")

        df = df.copy()
        df.sort_values("Date", inplace=True)

        # Log returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Rolling volatility (20-day annualized)
        df["rolling_volatility"] = (
            df["log_return"].rolling(window=20).std() * np.sqrt(252)
        )

        # Moving averages
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma50"] = df["Close"].rolling(50).mean()

        # RSI (14)
        df["rsi"] = ta.momentum.RSIIndicator(
            close=df["Close"],
            window=14
        ).rsi()

        # Volume percentage change
        df["volume_change"] = df["Volume"].pct_change()

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

        # Next-day targets
# Predict next-day return instead of raw price
        df["target_return"] = df["log_return"].shift(-1)

# Predict next-day volatility
        df["target_volatility"] = df["rolling_volatility"].shift(-1)


        # Regime labeling
        # Bullish = MA20 > MA50
        df["regime"] = np.where(df["ma20"] > df["ma50"], 1, 0)

        # Replace infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaNs
        df.dropna(inplace=True)

        logging.info(f"Feature engineering complete. Final rows: {len(df)}")

        return df

    def save_processed(self, df: pd.DataFrame):
        """
        Save processed dataset.
        """
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logging.info(f"Processed data saved to {self.processed_path}")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline.
        """
        df = self.compute_features(df)
        self.save_processed(df)
        return df
