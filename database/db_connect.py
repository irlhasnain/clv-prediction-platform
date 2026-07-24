import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__),'clv.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables():
    conn = get_connection()
    with open('database/schema.sql','r') as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close
    print("Table created successfully")

if __name__ == "__main__":
    create_tables()