import os.path
import time

import pandas as pd
import yfinance as yf
from yfinance import exceptions

from pandas_datareader import data as pdr


def need_get(df):
    for index, row in df.iterrows():
        mark = row["Mark"]
        if mark == 1:
            continue
        stock_name = str(row["Stock_name"])
        print("now:", stock_name, " -- ", index + 1, r"/", len(df))
        df.at[index, 'Mark'] = get_price(stock_name)
        df.to_csv("lists/" + list_file_path, index=False)
        time.sleep(0.1)
        yield df


def get_price(ticker):
    # tickerStrings = ['AAPL', 'MSFT']
    yf.pdr_override()
    attempts = 0
    while attempts < 3:
        try:
            timer = time.time()
            data = pdr.get_data_yahoo(ticker, start="1970-01-01", end="2024-02-04")
            data.to_csv("yahoo/" + ticker + ".csv")
            print("Used:", time.time() - timer)
            flag = 1
            return flag
        except exceptions.YFinanceException:
            print("Failed to download")
            attempts += 1
            continue
    if attempts == 3:
        flag = 0
        return flag


if __name__ == "__main__":
    list_len = 0
    inits = input("from which lists:")
    for init in inits:
        list_file_path = 'list_' + init + '.csv'
        list_df = pd.read_csv("lists/" + list_file_path, encoding="utf-8")
        list_len = list_len + len(list_df)
        while not list_df['Mark'].isin([1]).all():
            print("need get")
            list_df = next(need_get(list_df))
        print("list_", init, "completed", "len:", len(list_df))
    print("total:", list_len)
