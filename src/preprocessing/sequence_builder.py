"""
Sequence Builder Module
Converts tabular time series into windowed sequences
for CNN-LSTM architecture.
"""

import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SequenceBuilder:
    """
    Converts flat time series arrays into sequences.
    """

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length

    def create_sequences(self, X, y_regression, y_classification):
        """
        Create rolling window sequences.

        Returns:
            X_seq: (samples, seq_len, features)
            y_reg_seq: (samples, 2)
            y_class_seq: (samples,)
        """

        X_seq = []
        y_reg_seq = []
        y_class_seq = []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_reg_seq.append(y_regression[i])
            y_class_seq.append(y_classification[i])

        X_seq = np.array(X_seq)
        y_reg_seq = np.array(y_reg_seq)
        y_class_seq = np.array(y_class_seq)

        logging.info(f"Sequences created: {X_seq.shape}")

        return X_seq, y_reg_seq, y_class_seq

    def run(self, X_train, X_test, y_train, y_test,
            y_reg_train, y_reg_test):

        X_train_seq, y_train_seq, y_train_class = self.create_sequences(
            X_train, y_train, y_reg_train
        )

        X_test_seq, y_test_seq, y_test_class = self.create_sequences(
            X_test, y_test, y_reg_test
        )

        return (
            X_train_seq,
            X_test_seq,
            y_train_seq,
            y_test_seq,
            y_train_class,
            y_test_class
        )
