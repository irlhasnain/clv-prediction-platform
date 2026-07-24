import pandas as pd 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path, encoding='latin-1')

    df = df.dropna()
    df = df.drop_duplicates()

    df.columns = [c.strip().lower().replace(' ','_').replace('-','_') for c in df.columns]

    df['order_date'] = pd.to_datetime(df['order_date'], errors = 'coerce')
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors = 'coerce')

    df = df.dropna(subset=['customer_name', 'product_name'])

    df.to_csv(output_path, index= False)
    print(f"Cleaned data saved: {df.shape}")
    return df 

if __name__ == "__main__":
    clean_data("data/raw/Sample - Superstore.csv", "data/processed/cleaned_data.csv")