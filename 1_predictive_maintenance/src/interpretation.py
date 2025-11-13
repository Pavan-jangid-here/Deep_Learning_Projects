import shap
import numpy as np
from tensorflow.keras.models import load_model

data = np.load('data/train_seq.npz')
X = data['X']                   # shape: (samples, seq_len, n_features)
X_flat = X.reshape((X.shape[0], -1))  # flatten time series for KernelExplainer (works on tabular data)

model = load_model('data/lstm_cnn_model.h5')

def model_predict(input_array):
    # reshape for model (batch_size, seq_len, n_features)
    reshaped = input_array.reshape((-1, X.shape[1], X.shape[2]))
    return model.predict(reshaped)

explainer = shap.KernelExplainer(model_predict, X_flat[:100])  # use a small subset for background
shap_values = explainer.shap_values(X_flat[:10])  # explain first 10 samples
shap.summary_plot(shap_values, features=X_flat[:10], feature_names=[f'feat_{i}' for i in range(X_flat.shape[1])])
