import pandas as pd 

FEATURES = ["frequency", "recency", "customer_age_days", "avg_order_value"]

df = pd.read_csv("data/processed/customer_features.csv")[FEATURES]

reference = df.sample(frac=0.5, random_state=42)
current = df.drop(reference.index).copy()

current["avg_order_value"] = current["avg_order_value"] * 1.5
current["recency"] = current["recency"] * 1.3

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html("docs/drift_report.html")
except ImportError:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html("docs/drift_report.html")

print("Drift report saved to docs/drift_report.html")
