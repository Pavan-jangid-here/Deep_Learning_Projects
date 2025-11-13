import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def rolling_and_lag_features(df, sensors, roll_window=3):
    for s in sensors:
        df[f'{s}_roll_mean'] = df.groupby('unit')[s].rolling(roll_window).mean().reset_index(drop=True)
        df[f'{s}_roll_std']  = df.groupby('unit')[s].rolling(roll_window).std().reset_index(drop=True)
        df[f'{s}_lag1']      = df.groupby('unit')[s].shift(1)
    return df

def outlier_removal(df, sensors, thresh=20):
    Z = (df[sensors] - df[sensors].mean()) / df[sensors].std()
    mask = (np.abs(Z) < thresh).all(axis=1)
    return df[mask]

if __name__ == '__main__':
    df = pd.read_csv('data/nasa_prepared.csv')
    sensors = [f'sensor_{i}' for i in range(1,22)]
    df = rolling_and_lag_features(df, sensors)
    df = df.fillna(method="bfill")   # Handle NaNs from rolling/lag
    df = outlier_removal(df, sensors)
    if df.shape[0] == 0:
        raise ValueError("No data left after outlier removal. Try a higher threshold or skip this step.")
    scaler = MinMaxScaler()
    feature_cols = [col for col in df.columns if col not in ['unit','cycle','RUL']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df.to_csv('data/nasa_engineered.csv', index=False)
