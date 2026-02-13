"""
Return-Based Inference Engine
"""

import torch
import joblib
import numpy as np
import torch.nn.functional as F

from src.ingestion.data_loader import SensexDataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.model.cnn_lstm_model import CNNLSTMModel


class Predictor:

    def __init__(self, sequence_length=30):

        self.sequence_length = sequence_length

        self.feature_scaler = joblib.load("models/saved/feature_scaler.pkl")
        self.return_scaler = joblib.load("models/saved/return_scaler.pkl")
        self.vol_scaler = joblib.load("models/saved/vol_scaler.pkl")

        self.model = CNNLSTMModel(
            input_size=15,
            sequence_length=sequence_length
        )

        self.model.load_state_dict(
            torch.load("models/saved/best_model.pth",
                       map_location="cpu")
        )

        self.model.eval()

    def predict_next_day(self):

        loader = SensexDataLoader(years=12)
        raw_df = loader.run()

        fe = FeatureEngineer()
        df = fe.compute_features(raw_df)

        latest_close = df["Close"].iloc[-1]

        feature_columns = [
            "Close","log_return","rolling_volatility",
            "ma20","ma50","rsi","volume_change",
            "close_lag_1","close_lag_2","close_lag_3",
            "close_lag_5","return_lag_1","return_lag_2",
            "return_lag_3","return_lag_5"
        ]

        X = df[feature_columns].values
        X_scaled = self.feature_scaler.transform(X)

        latest_seq = X_scaled[-self.sequence_length:]
        latest_seq = np.expand_dims(latest_seq, axis=0)

        X_tensor = torch.tensor(latest_seq, dtype=torch.float32)

        with torch.no_grad():
            return_pred, vol_pred, regime_logits = self.model(X_tensor)

        return_pred = return_pred.numpy()
        vol_pred = vol_pred.numpy()

        # Inverse scaling
        predicted_return = float(
            self.return_scaler.inverse_transform(return_pred)[0][0]
        )

        predicted_vol = float(
            self.vol_scaler.inverse_transform(vol_pred)[0][0] * 100
        )

        # Reconstruct price
        predicted_price = float(
            latest_close * np.exp(predicted_return)
        )

        regime_probs = F.softmax(
            regime_logits,
            dim=1
        ).numpy()

        regime_label = int(np.argmax(regime_probs, axis=1)[0])
        confidence = float(np.max(regime_probs))

        regime_text = "Bullish" if regime_label == 1 else "Bearish"

        return {
            "predicted_sensex_level": predicted_price,
            "predicted_volatility_percent": predicted_vol,
            "regime": regime_text,
            "confidence": confidence
        }
