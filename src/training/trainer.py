"""
Training Engine
Sensex CNN-LSTM Forecasting Engine
"""

import os
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Trainer:
    """
    Handles training loop, loss computation, and checkpointing.
    """

    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-5,
        alpha=1.0,   # price weight
        beta=1.0,    # volatility weight
        gamma=0.5,   # regime weight
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.checkpoint_path = "models/saved/best_model.pth"

    def compute_loss(
        self,
        price_pred,
        vol_pred,
        regime_pred,
        price_true,
        vol_true,
        regime_true
    ):

        loss_price = self.mse(price_pred, price_true)
        loss_vol = self.mse(vol_pred, vol_true)
        loss_regime = self.ce(regime_pred, regime_true)

        total_loss = (
            self.alpha * loss_price +
            self.beta * loss_vol +
            self.gamma * loss_regime
        )

        return total_loss

    def train(
        self,
        X_train,
        y_train,
        y_train_class,
        epochs=50,
        batch_size=32,
        patience=10
    ):

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_train_class = torch.tensor(
            y_train_class,
            dtype=torch.long
        ).to(self.device)

        dataset_size = X_train.shape[0]
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):

            self.model.train()
            epoch_loss = 0.0

            for i in range(0, dataset_size, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                y_class_batch = y_train_class[i:i+batch_size]

                price_true = y_batch[:, 0:1]
                vol_true = y_batch[:, 1:2]

                self.optimizer.zero_grad()

                price_pred, vol_pred, regime_pred = self.model(X_batch)

                loss = self.compute_loss(
                    price_pred,
                    vol_pred,
                    regime_pred,
                    price_true,
                    vol_true,
                    y_class_batch
                )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (dataset_size // batch_size)

            logging.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {epoch_loss:.6f}"
            )

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0

                os.makedirs("models/saved", exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training complete.")

def evaluate(self, X_test, y_test, y_test_class):
    """
    Evaluate the trained model on test data.
    """
    from src.evaluation.evaluator import Evaluator

    # Load best model
    self.model.load_state_dict(
        torch.load(self.checkpoint_path)
    )

    evaluator = Evaluator(self.model)

    results = evaluator.evaluate(
        X_test,
        y_test,
        y_test_class
    )

    return results