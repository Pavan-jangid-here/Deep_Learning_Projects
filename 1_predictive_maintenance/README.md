# Predictive Maintenance for Industrial Equipment (Time Series LSTM/CNN)

## Overview

This project predicts Remaining Useful Life (RUL) for industrial equipment using deep learning (LSTM/CNN) on time series sensor data. The solution covers all steps from data cleaning, feature engineering, model development, evaluation, interpretation, and real-time dashboard deployment with Streamlit.

## Features

- Data pipeline: Cleaning, missing value handling, rolling/lags, outlier removal  
- Deep learning: LSTM/CNN model for RUL regression  
- Interpretability: SHAP feature importance analysis  
- Interactive dashboard: Real-time upload and prediction with Streamlit

## Dataset

**Dataset Name:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Turbofan Engine Degradation Simulation Dataset - FD001

- Source: NASA Prognostics Data Repository  
- Description: Simulated turbofan engine degradation data with sensor readings over operational cycles  
- File: `train_FD001.txt` (space-separated values, no header)  
- Columns: `unit_id`, `cycle`, operating conditions (3), sensor measurements (21)

**Sample data format:**
1 1 0.0007 0.0005 21.2030 518.67 641.82 1589.70 1400.60 14.62 21.61 554.37 2388.06 9046.19 1.30 47.47 521.66 2388.42 8138.76 8.42 0.03 392.00 2388.00
1 2 0.0010 0.0003 21.2851 518.67 642.15 1591.82 1403.14 14.62 21.61 553.91 2388.04 9046.19 1.30 47.49 522.28 2388.42 8131.49 8.42 0.02 392.00 2388.00


## Project Structure
1_predictive_maintenance/
├── data/
│ ├── train_FD001.txt
│ └── sample_input.csv
├── src/
│ ├── data_prep.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── evaluation.py
│ └── interpretation.py
├── requirements.txt
├── README.md
└── streamlit_app.py



## Installation & Setup

1. Clone repository:
    ```
    git clone https://github.com/pavan-jangid-here/Deep_Learning_Projects.git
    cd Deep_Learning_Projects/1_predictive_maintenance
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Download dataset:
    - Place `train_FD001.txt` in the `data/` folder  (Already Downloaded)
    - Dataset available from NASA Prognostics Data Repository

## Usage

### Step 1: Data Preparation
python src/data_prep.py



### Step 2: Feature Engineering
python src/feature_engineering.py



### Step 3: Model Training
python src/model.py



### Step 4: Model Evaluation
python src/evaluation.py



### Step 5: SHAP Interpretation
python src/interpretation.py



### Step 6: Launch Dashboard
streamlit run app/streamlit_app.py



## Model Architecture

- Input: 30-cycle sequences of sensor readings  
- CNN Layer: 32 filters, kernel size 3  
- LSTM Layer: 64 units  
- Dense Layers: 32 units → 1 output (RUL prediction)  
- Optimization: Adam optimizer, MSE loss


## Dashboard Features

- Upload new sensor data (CSV format)  
- Real-time RUL prediction  
- Maintenance recommendation timeline  
- Prediction confidence visualization

## Sample Input Format

unit,cycle,op_1,op_2,op_3,sensor_1,sensor_2,sensor_3,...,sensor_21
1,1,0.0007,0.0005,21.2030,518.67,641.82,1589.7,...,2388.0
1,2,0.0010,0.0003,21.2851,518.67,642.15,1591.8,...,2388.0



## Technologies Used

- Python 3.8+  
- TensorFlow/Keras - Deep learning framework  
- Pandas/NumPy - Data manipulation  
- Scikit-learn - Preprocessing and metrics  
- SHAP - Model interpretability  
- Streamlit - Interactive dashboard  
- Matplotlib/Seaborn - Visualization

## Future Enhancements

- Multi-engine type support (FD002, FD003, FD004)  
- Real-time streaming data integration  
- Advanced anomaly detection  
- Cloud deployment (AWS/Azure)

## Contact

Created by Pavan Jangid | [LinkedIn](https://linkedin.com/in/pavan-jangid-big-data-analyst) | [Email](mailto:Pavan5.jan@gmail.com)

## License

This project is licensed under the MIT License.

## Acknowledgments

- NASA Prognostics Center of Excellence for the C-MAPSS dataset  
- TensorFlow and Streamlit development teams