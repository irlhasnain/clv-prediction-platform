# 🎯 Customer Lifetime Value (CLV) Prediction Platform

An end-to-end machine learning system that predicts customer lifetime value using RFM-based features, and serves real-time predictions through a REST API.

## 🎯 Project Overview

Unlike traditional batch-analytics projects, this system is built as a **production-style ML pipeline**:
- **Database Design**: Normalized SQLite schema (customers, products, orders, order_items)
- **ETL Pipeline**: Automated cleaning and loading of raw transaction data
- **Feature Engineering**: RFM (Recency, Frequency, Monetary) feature extraction via SQL
- **ML Model**: Random Forest regressor predicting customer lifetime value, benchmarked against a formula-based baseline
- **Real-Time API**: FastAPI service that serves live CLV predictions for new customer inputs
- **Testing**: Unit tests covering API health and prediction endpoints

## 🏗️ Architecture

## 📁 Project Structure
```
clv-prediction-platform/
├── data/ # Raw and processed datasets
├── database/ # DB connection and schema
├── etl/ # Data cleaning and loading scripts
├── features/ # RFM feature engineering
├── models/ # Model training and saved model
├── api/ # FastAPI serving app
├── notebooks/ # Experimentation notebooks
├── tests/ # Unit tests
└── requirements.txt
```
## 🔧 Tech Stack

- **Language**: Python
- **Database**: SQLite
- **ML**: scikit-learn (Random Forest)
- **API**: FastAPI, Pydantic, Uvicorn
- **Testing**: Pytest, httpx

## 📈 Key Results

- Prophet... *(if applicable, otherwise remove)*
- Random Forest CLV model achieved an MAE of **[apna number daalo]**, compared to a formula-based baseline MAE of **[apna number daalo]**
- API responds to prediction requests in real time via a single POST call

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/irlhasnain/clv-prediction-platform.git
cd clv-prediction-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Set up database and run ETL
python -m database.db_connect
python -m etl.clean_data
python -m etl.load_data

# Build features and train model
python -m features.build_feature
python -m models.train_model

# Run tests
pytest tests/

# Start the API
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

## 📡 API Usage Example

**Endpoint**: `POST /predict`

**Request:**
```json
{
  "frequency": 5,
  "recency": 30,
  "customer_age_days": 365,
  "avg_order_value": 150
}
```

**Response:**
```json
{
  "predicted_clv": 2450.75
}
```

## 🔗 Live Demo

[API Documentation](https://clv-prediction-platform-t672.onrender.com/docs)

## 📝 License

This project is licensed under the MIT License.
