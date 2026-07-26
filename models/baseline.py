import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np 

def baseline_prediction():
    df = pd.read_csv('data/processed/customer_features.csv')

    df['baseline_clv'] = df['avg_order_value'] * df['frequency']

    mae = mean_absolute_error(df['monetary'], df['baseline_clv'])
    rmse = np.sqrt(mean_squared_error(df['monetary'],df['baseline_clv']))

    print(f"Baseline MAE : {mae:.2f}")
    print(f"Baseline RMSE : {rmse:.2f}")

    return mae, rmse

if __name__ == "__main__":
    baseline_prediction()
