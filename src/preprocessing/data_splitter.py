"""
Data Splitting and Scaling Module
Return-based Target Version
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataSplitter:

    def __init__(self, test_size: float = 0.2):

        self.test_size = test_size
        self.feature_scaler = MinMaxScaler()
        self.return_scaler = MinMaxScaler()
        self.vol_scaler = MinMaxScaler()

    def split(self, df: pd.DataFrame):

        split_index = int(len(df) * (1 - self.test_size))

        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        logging.info(f"Train size: {len(train_df)}")
        logging.info(f"Test size: {len(test_df)}")

        return train_df, test_df

    def prepare_features_targets(self, df: pd.DataFrame):

        feature_columns = [
            "Close",
            "log_return",
            "rolling_volatility",
            "ma20",
            "ma50",
            "rsi",
            "volume_change",
            "close_lag_1",
            "close_lag_2",
            "close_lag_3",
            "close_lag_5",
            "return_lag_1",
            "return_lag_2",
            "return_lag_3",
            "return_lag_5"
        ]

        X = df[feature_columns].values

        y_return = df["target_return"].values.reshape(-1, 1)
        y_vol = df["target_volatility"].values.reshape(-1, 1)

        y_regime = df["regime"].values

        return X, y_return, y_vol, y_regime

    def scale(self,
              X_train, X_test,
              y_train_return, y_test_return,
              y_train_vol, y_test_vol):

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        y_train_return_scaled = self.return_scaler.fit_transform(y_train_return)
        y_test_return_scaled = self.return_scaler.transform(y_test_return)

        y_train_vol_scaled = self.vol_scaler.fit_transform(y_train_vol)
        y_test_vol_scaled = self.vol_scaler.transform(y_test_vol)

        y_train_scaled = np.hstack([
            y_train_return_scaled,
            y_train_vol_scaled
        ])

        y_test_scaled = np.hstack([
            y_test_return_scaled,
            y_test_vol_scaled
        ])

        os.makedirs("models/saved", exist_ok=True)

        joblib.dump(self.feature_scaler, "models/saved/feature_scaler.pkl")
        joblib.dump(self.return_scaler, "models/saved/return_scaler.pkl")
        joblib.dump(self.vol_scaler, "models/saved/vol_scaler.pkl")

        logging.info("Scalers saved (return-based).")

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def run(self, df: pd.DataFrame):

        train_df, test_df = self.split(df)

        X_train, y_train_return, y_train_vol, y_reg_train = \
            self.prepare_features_targets(train_df)

        X_test, y_test_return, y_test_vol, y_reg_test = \
            self.prepare_features_targets(test_df)

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = \
            self.scale(
                X_train, X_test,
                y_train_return, y_test_return,
                y_train_vol, y_test_vol
            )

        return (
            X_train_scaled,
            X_test_scaled,
            y_train_scaled,
            y_test_scaled,
            y_reg_train,
            y_reg_test
        )
