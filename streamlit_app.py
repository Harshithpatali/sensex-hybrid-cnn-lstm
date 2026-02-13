import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score

from src.inference.predictor import Predictor
from src.ingestion.data_loader import SensexDataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.sequence_builder import SequenceBuilder
from src.model.cnn_lstm_model import CNNLSTMModel
import joblib


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Sensex Hybrid CNN-LSTM Engine",
    layout="wide"
)

st.title("🇮🇳 Sensex Hybrid CNN-LSTM Forecasting Engine")
st.markdown("---")


# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def get_next_trading_day(last_date):
    next_day = last_date + timedelta(days=1)
    if next_day.weekday() == 5:
        next_day += timedelta(days=2)
    if next_day.weekday() == 6:
        next_day += timedelta(days=1)
    return next_day


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    loader = SensexDataLoader(years=3)
    return loader.run()

df = load_data()

current_price = df["Close"].iloc[-1]
last_available_date = df["Date"].iloc[-1]
prediction_date = get_next_trading_day(last_available_date)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
@st.cache_resource
def load_predictor():
    return Predictor()

predictor = load_predictor()
prediction = predictor.predict_next_day()

pred_price = prediction["predicted_sensex_level"]
pred_vol = prediction["predicted_volatility_percent"]
regime = prediction["regime"]
confidence = prediction["confidence"]

price_delta = pred_price - current_price
pred_return_pct = (price_delta / current_price) * 100

st.info(
    f"Prediction for close on {prediction_date.date()} "
    f"(Last data: {last_available_date.date()})"
)

# ---------------------------------------------------
# Top Metrics
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Current Close", f"{current_price:,.0f}")
col2.metric("Predicted Close", f"{pred_price:,.0f}",
            delta=f"{price_delta:,.0f}")
col3.metric("Predicted Return", f"{pred_return_pct:.2f}%")

st.markdown("---")

col4, col5 = st.columns(2)
col4.metric("Predicted Volatility", f"{pred_vol:.2f}%")

if regime == "Bullish":
    col5.success(f"Regime: {regime}")
else:
    col5.error(f"Regime: {regime}")

st.write(f"Confidence: {confidence:.2%}")
st.progress(float(confidence))

st.markdown("---")


# ---------------------------------------------------
# Backtest + Analytics
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def generate_backtest():

    loader = SensexDataLoader(years=3)
    raw_df = loader.run()

    fe = FeatureEngineer()
    df = fe.compute_features(raw_df)

    splitter = DataSplitter(test_size=0.2)

    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = splitter.run(df)

    seq_builder = SequenceBuilder(sequence_length=30)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _, _ = seq_builder.run(
        X_train, X_test, y_train, y_test, y_reg_train, y_reg_test
    )

    model = CNNLSTMModel(
        input_size=X_train_seq.shape[2],
        sequence_length=30
    )

    model.load_state_dict(
        torch.load("models/saved/best_model.pth",
                   map_location="cpu")
    )

    model.eval()

    X_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

    with torch.no_grad():
        return_pred, _, _ = model(X_tensor)

    return_scaler = joblib.load("models/saved/return_scaler.pkl")

    predicted_returns = return_scaler.inverse_transform(
        return_pred.numpy()
    ).flatten()

    actual_returns = df["target_return"].iloc[-len(predicted_returns):].values

    test_close = df["Close"].iloc[-len(predicted_returns):].values

    predicted_prices = []
    price = test_close[0]

    for r in predicted_returns:
        price = price * np.exp(r)
        predicted_prices.append(price)

    predicted_prices = np.array(predicted_prices)

    return test_close[:len(predicted_prices)], predicted_prices, actual_returns, predicted_returns


actual_prices, predicted_prices, actual_returns, predicted_returns = generate_backtest()


# ---------------------------------------------------
# RMSE & R²
# ---------------------------------------------------
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

col6, col7 = st.columns(2)
col6.metric("RMSE (Test)", f"{rmse:,.0f}")
col7.metric("R² (Test)", f"{r2:.3f}")

st.markdown("---")


# ---------------------------------------------------
# Predicted vs Actual Chart
# ---------------------------------------------------
st.subheader("Predicted vs Actual")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=actual_prices, name="Actual"))
fig1.add_trace(go.Scatter(y=predicted_prices, name="Predicted", line=dict(dash="dash")))
fig1.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig1, use_container_width=True)


# ---------------------------------------------------
# Rolling Error Chart
# ---------------------------------------------------
rolling_error = pd.Series(actual_prices - predicted_prices).rolling(20).mean()

st.subheader("Rolling Prediction Error (20-day MA)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=rolling_error, name="Rolling Error"))
fig2.update_layout(template="plotly_white", height=350)
st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------
# Directional Accuracy
# ---------------------------------------------------
direction_actual = np.sign(actual_returns)
direction_pred = np.sign(predicted_returns)

directional_accuracy = np.mean(direction_actual == direction_pred)

st.metric("Directional Accuracy", f"{directional_accuracy:.2%}")

st.markdown("---")


# ---------------------------------------------------
# Cumulative Strategy Return
# ---------------------------------------------------
strategy_returns = np.where(predicted_returns > 0,
                            actual_returns,
                            -actual_returns)

cum_strategy = np.cumprod(1 + strategy_returns)
cum_market = np.cumprod(1 + actual_returns)

st.subheader("Cumulative Return Strategy")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=cum_market, name="Buy & Hold"))
fig3.add_trace(go.Scatter(y=cum_strategy, name="Model Strategy"))
fig3.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------
# Probability Cone (1-day projection)
# ---------------------------------------------------
st.subheader("Probability Cone (1-Day Projection)")

vol_daily = pred_vol / 100 / np.sqrt(252)

upper = current_price * np.exp(vol_daily)
lower = current_price * np.exp(-vol_daily)

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=[0, 1],
    y=[current_price, upper],
    name="Upper Band"
))

fig4.add_trace(go.Scatter(
    x=[0, 1],
    y=[current_price, lower],
    name="Lower Band"
))

fig4.update_layout(
    template="plotly_white",
    height=350,
    xaxis_title="Today → Tomorrow",
    yaxis_title="Projected Range"
)

st.plotly_chart(fig4, use_container_width=True)


st.markdown("---")
st.caption("Hybrid CNN-LSTM Return-Based Forecasting Engine | Quant Analytics Dashboard")
