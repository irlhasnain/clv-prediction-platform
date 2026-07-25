import pandas as pd
import sys
sys.path.append('.')
from database.db_connect import get_connection

def build_customer_features():
    conn = get_connection()

    query = """
    SELECT 
        c.customer_id,
        COUNT(DISTINCT o.order_id) as frequency,
        SUM(oi.sales) as monetary,
        julianday('now') - julianday(MAX(o.order_date)) as recency,
        julianday(MAX(o.order_date)) - julianday(MIN(o.order_date)) as customer_age_days,
        AVG(oi.sales) as avg_order_value
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id
    """

    feature_df = pd.read_sql_query(query, conn)
    conn.close()

    feature_df['customer_age_days'] = feature_df['customer_age_days'].fillna(0)
    return feature_df

if __name__ == "__main__":
    df = build_customer_features()
    print(df.head())
    print(f"Total customer with features : {df.shape[0]}")

    df.to_csv('data/processed/customer_features.csv',index=False)
    print("Feature Saved!")