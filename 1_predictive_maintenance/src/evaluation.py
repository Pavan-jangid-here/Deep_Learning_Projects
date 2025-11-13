import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    data = np.load('data/train_seq.npz')
    X, y = data['X'], data['y']
    model = load_model('data/lstm_cnn_model.h5')
    y_pred = model.predict(X).flatten()
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f'Full Test RMSE: {rmse:.2f}, MAE: {mae:.2f}')
