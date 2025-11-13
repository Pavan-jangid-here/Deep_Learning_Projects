import numpy as np
import pandas as pd

def gen_sequences(df, seq_len=30, features=None, target='RUL'):
    X, y = [], []
    units = df['unit'].unique()
    for unit in units:
        df_u = df[df['unit'] == unit]
        for i in range(len(df_u) - seq_len):
            X.append(df_u.iloc[i:i+seq_len][features].values)
            y.append(df_u.iloc[i+seq_len][target])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    df = pd.read_csv('data/nasa_prepared.csv')
    features = [col for col in df.columns if col not in ['unit','cycle','RUL']]
    X, y = gen_sequences(df, seq_len=30, features=features)
    np.savez('data/train_seq.npz', X=X, y=y)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train_model():
    data = np.load('data/train_seq.npz')
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(seq_len, n_feat)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    model.save('data/lstm_cnn_model.h5')
    return model, X_test, y_test

if __name__ == '__main__':
    train_model()
