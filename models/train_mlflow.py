import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

FEATURE = ["frequency", "recency", "customer_age_days", "avg_order_value"]
TARGET = "monetary"

df = pd.read_csv("data/processed/customer_features.csv")
X = df[FEATURE]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

experiments = {
    "ridge": (Ridge(alpha=1.0), {"alpha": 1.0}),
    "random_forest_100":(
        RandomForestRegressor(n_estimators=100, random_state=42),
        {"n_estimators": 100},
    ),
    "random_forest_300":(
        RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
        {"n_estimators": 300, "max_depth": 10},
    ),
    "gradient_boosting":(
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": 100, "learning_rate": 0.1},
    ),
}

mlflow.set_experiment("customer_lifetime_value_prediction")

baseline_pred = X_test["frequency"] * X_test["avg_order_value"]
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

with mlflow.start_run(run_name="formula_baseline"):
    mlflow.log_metrics({"mae": baseline_mae, "rmse": baseline_rmse, "r2": baseline_r2})
print(f"Baseline: MAE: {baseline_mae:.2f}, RMSE: {baseline_rmse:.2f}, R2: {baseline_r2:.3f}")

for name, (model, params) in experiments.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"{name}: MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae = -cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, cv=cv, scoring="neg_mean_absolute_error",
)
print(f"RF 5-fold CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.0f}")