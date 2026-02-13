"""
Evaluation Engine
Sensex CNN-LSTM Forecasting Engine
"""

import torch
import numpy as np
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)
import joblib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Evaluator:
    """
    Evaluates trained model on test data.
    """

    def __init__(self, model, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()

        # Load target scaler for inverse transform
        self.target_scaler = joblib.load(
            "models/saved/target_scaler.pkl"
        )

    def evaluate(self, X_test, y_test, y_test_class):

        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            price_pred, vol_pred, regime_pred = self.model(X_test)

        price_pred = price_pred.cpu().numpy()
        vol_pred = vol_pred.cpu().numpy()
        regime_pred = regime_pred.cpu().numpy()

        # Combine regression outputs
        reg_pred_scaled = np.hstack([price_pred, vol_pred])

        # Inverse scaling
        reg_pred = self.target_scaler.inverse_transform(reg_pred_scaled)
        reg_true = self.target_scaler.inverse_transform(y_test)

        # Regression metrics
        rmse = np.sqrt(mean_squared_error(reg_true, reg_pred))
        mae = mean_absolute_error(reg_true, reg_pred)
        r2 = r2_score(reg_true, reg_pred)

        # Classification
        regime_labels = np.argmax(regime_pred, axis=1)
        accuracy = accuracy_score(y_test_class, regime_labels)
        cm = confusion_matrix(y_test_class, regime_labels)

        results = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Regime Accuracy": accuracy,
            "Confusion Matrix": cm
        }

        return results
