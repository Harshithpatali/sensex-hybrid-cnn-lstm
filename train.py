import torch

from src.ingestion.data_loader import SensexDataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.sequence_builder import SequenceBuilder
from src.model.cnn_lstm_model import CNNLSTMModel
from src.training.trainer import Trainer


if __name__ == "__main__":

    # 1️⃣ Ingestion
    loader = SensexDataLoader(years=12)
    raw_df = loader.run()

    # 2️⃣ Feature Engineering
    fe = FeatureEngineer()
    processed_df = fe.run(raw_df)

    # 3️⃣ Split & Scale
    splitter = DataSplitter(test_size=0.2)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_reg_train,
        y_reg_test
    ) = splitter.run(processed_df)

    # 4️⃣ Sequence Builder
    seq_builder = SequenceBuilder(sequence_length=30)

    (
        X_train_seq,
        X_test_seq,
        y_train_seq,
        y_test_seq,
        y_train_class,
        y_test_class
    ) = seq_builder.run(
        X_train,
        X_test,
        y_train,
        y_test,
        y_reg_train,
        y_reg_test
    )

    # 5️⃣ Model
    model = CNNLSTMModel(
        input_size=X_train_seq.shape[2],
        sequence_length=30
    )

    # 6️⃣ Trainer
    trainer = Trainer(model)

    trainer.train(
        X_train_seq,
        y_train_seq,
        y_train_class,
        epochs=30,
        batch_size=32
    )
from src.inference.predictor import Predictor

predictor = Predictor()
output = predictor.predict_next_day()

print("\nNext Day Forecast:")
print(output)
