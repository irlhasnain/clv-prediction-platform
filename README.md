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
