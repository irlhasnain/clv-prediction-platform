import pandas as pd 
from database.db_connect import get_connection

def load_customers(df, conn):
    customer_df = df[['customer_id', 'customer_name', 'segment', 'region']].drop_duplicates()
    customer_df.to_sql('customers', conn, if_exists='replace', index=False)
    print(f"Loaded {len(customer_df)} customers")

def load_products(df, conn):
    product_df = df[['product_id', 'product_name', 'category', 'sub_category']].drop_duplicates()
    product_df.to_sql('products', conn, if_exists='replace', index=False)
    print(f"Loaded {len(product_df)} products")

def load_orders(df, conn):
    order_df = df[['order_id', 'customer_id', 'order_date', 'ship_date']].drop_duplicates()
    order_df.to_sql('orders', conn, if_exists='replace', index=False)
    print(f"Loaded {len(order_df)} orders")

def load_order_items(df, conn):
    order_item_df = df[['order_id', 'product_id', 'quantity', 'sales', 'profit']]
    order_item_df.to_sql('order_items', conn, if_exists='replace', index=False)
    print(f"Loaded {len(order_item_df)}  order items")

def run_etl():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    conn = get_connection()
    load_customers(df, conn)
    load_products(df, conn)
    load_orders(df, conn)
    load_order_items(df, conn)
    conn.close()
    print("ETL process completed successfully!")

if __name__ == "__main__":
    run_etl()
