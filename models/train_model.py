import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

def train_clv_model():
    df = pd.read_csv('data/processed/customer_features.csv')

    X = df[['frequency', 'recency', 'customer_age_days','avg_order_value']]
    Y = df['monetary']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    