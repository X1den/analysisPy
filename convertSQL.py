import sqlite3
import pandas as pd
import os

def csv_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def df_to_sql(df: pd.DataFrame, db_name: str, table_name: str):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def sql_to_df(db_name: str, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == '__main__':
    df = csv_to_df('data/titanic.csv')
    df_to_sql(df, 'data\mydata.db', 'titanic')

    query = "SELECT Sex, COUNT(*) as cnt FROM titanic Group by Sex"
    result = sql_to_df('data\mydata.db', query)
    print(result)