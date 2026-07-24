import pandas as pd 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path, encoding='latin-1')
