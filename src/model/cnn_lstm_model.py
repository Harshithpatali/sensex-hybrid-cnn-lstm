"""
Hybrid CNN-LSTM Model
Multi-Task: Price, Volatility, Regime
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN + LSTM for time series forecasting.
    Multi-head outputs:
        - price (regression)
        - volatility (regression)
        - regime (classification)
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        cnn_channels: int = 64,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CNNLSTMModel, self).__init__()

        self.sequence_length = sequence_length

        # 1D CNN Block
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # Output Heads
        self.price_head = nn.Linear(lstm_hidden_size, 1)
        self.vol_head = nn.Linear(lstm_hidden_size, 1)
        self.regime_head = nn.Linear(lstm_hidden_size, 2)

    def forward(self, x):
        """
        Forward pass.

        x shape: (batch_size, seq_len, features)
        """

        # Convert to (batch, features, seq_len) for CNN
        x = x.permute(0, 2, 1)

        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Convert back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_output = lstm_out[:, -1, :]

        last_output = self.dropout(last_output)

        # Heads
        price_output = self.price_head(last_output)
        vol_output = self.vol_head(last_output)
        regime_output = self.regime_head(last_output)

        return price_output, vol_output, regime_output
