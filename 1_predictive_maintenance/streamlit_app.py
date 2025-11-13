import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from src.data_prep import load_nasa_data, add_rul
from src.feature_engineering import rolling_and_lag_features
from src.model import gen_sequences

st.title("Industrial Equipment Predictive Maintenance: RUL Forecast")

uploaded_file = st.file_uploader("Upload 30-cycle sensor data (CSV, NASA format)", type="csv")
if uploaded_file:
    raw = pd.read_csv(uploaded_file)
    sensors = [f'sensor_{i}' for i in range(1,22)]
    raw = rolling_and_lag_features(raw, sensors)
    raw.fillna(method='bfill', inplace=True)
    features = [col for col in raw.columns if col not in ['unit', 'cycle', 'RUL']]
    X_new, _ = gen_sequences(raw, seq_len=30, features=features)
    model = tf.keras.models.load_model('data/lstm_cnn_model.h5')
    y_pred = model.predict(X_new).flatten()
    st.metric('Predicted RUL (mean)', int(np.mean(y_pred)))
    st.line_chart(y_pred)
    st.warning("Next maintenance recommended in approximately: {} cycles.".format(int(np.mean(y_pred))))
else:
    st.info("Upload your equipment sensor data (at least 30 cycles) for real-time prediction.")

# Demo output table for recruiters
st.header("Example Output")
df_demo = pd.DataFrame({
    "Cycle": np.arange(1,31),
    "Predicted RUL": np.clip(np.random.normal(70,6,30), 40,95)
})
st.dataframe(df_demo)
