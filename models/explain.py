import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

FEATURES = ["frequency", "recency", "customer_age_days", "avg_order_value"]

model = joblib.load("models/clv_model.pkl")
df = pd.read_csv("data/processed/customer_features.csv")
X = df[FEATURES]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("docs/shap_summary_plot.png", dpi=150)
plt.close()

shap.summary_plot(shap_values, X, feature_names=FEATURES, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("docs/shap_bar_plot.png", dpi=150)
plt.close()

mean_abs = pd.Series(abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print("Mean absolute SHAP values:")
print(mean_abs)