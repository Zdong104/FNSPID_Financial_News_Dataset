import os

import chardet
import numpy as np
import pandas as pd


def convert_to_utc(df, date_column):
    """
    将DataFrame中的日期列转换为UTC格式。
    """
    df[date_column] = pd.to_datetime(df[date_column])
    if df[date_column].dt.tz is None:  # 检查是否已经是UTC格式
        df[date_column] = df[date_column].dt.tz_localize('UTC')
    return df


def fill_missing_dates_with_log_decay(df, date_column, sentiment_column):
    # 确保日期格式正确
    df[date_column] = pd.to_datetime(df[date_column])

    # 创建完整日期范围
    date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())

    # 创建一个新的DataFrame，包含所有日期
    full_df = pd.DataFrame(date_range, columns=[date_column])
    full_df = pd.merge(full_df, df, on=date_column, how='left')

    # 初始化News_flag列
    full_df['News_flag'] = full_df[sentiment_column].notna().astype(int)

    # 使用对数衰减规则填充缺失的情感值
    last_valid_sentiment = None
    last_valid_date = None
    for i, row in full_df.iterrows():
        if pd.isna(row[sentiment_column]):
            if last_valid_sentiment is not None:
                days_since_last_valid = (row[date_column] - last_valid_date).days
                decayed_sentiment = last_valid_sentiment * (np.log(2) ** days_since_last_valid)
                full_df.at[i, sentiment_column] = decayed_sentiment
                full_df.at[i, 'News_flag'] = 0
        else:
            last_valid_sentiment = row[sentiment_column]
            last_valid_date = row[date_column]

    return full_df


def fill_missing_dates_with_exponential_decay(df, date_column, sentiment_column, sentiment_key_name, decay_rate=0.05):
    # 确保日期格式正确
    df[date_column] = pd.to_datetime(df[date_column])

    # 创建完整日期范围
    date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())

    # 创建一个新的DataFrame，包含所有日期
    full_df = pd.DataFrame(date_range, columns=[date_column])
    full_df = pd.merge(full_df, df, on=date_column, how='left')

    # 初始化News_flag列
    full_df['News_flag'] = full_df[sentiment_column].notna().astype(int)

    # 使用指数递减规则填充缺失的情感值
    last_valid_sentiment = None
    last_valid_date = None
    for i, row in full_df.iterrows():
        if pd.isna(row[sentiment_column]):
            if last_valid_sentiment is not None:
                days_since_last_valid = (row[date_column] - last_valid_date).days
                decayed_sentiment = 0
                if sentiment_key_name == "Sentiment_gpt":
                    decayed_sentiment = 3 + (last_valid_sentiment - 3) * np.exp(-decay_rate * days_since_last_valid)
                elif sentiment_key_name == "Sentiment_blob":
                    decayed_sentiment = last_valid_sentiment * np.exp(-decay_rate * days_since_last_valid)
                full_df.at[i, sentiment_column] = decayed_sentiment
                full_df.at[i, 'News_flag'] = 0
        else:
            last_valid_sentiment = row[sentiment_column]
            last_valid_date = row[date_column]

    return full_df


def integrate_data(stock_price_df, news_df, stock_price_csv_file, sentiment_key_name):
    # 创建原始DataFrame的副本
    stock_price_df_copy = stock_price_df.copy()
    news_df_copy = news_df.copy()
    # 2. 转换日期格式并排序
    stock_price_df_copy = convert_to_utc(stock_price_df_copy, 'Date')
    news_df_copy = convert_to_utc(news_df_copy, 'Date')

    stock_price_df_copy['Date'] = pd.to_datetime(stock_price_df_copy['Date'])
    news_df_copy['Date'] = pd.to_datetime(news_df_copy['Date'])

    # 将日期时间对齐到当天开始
    stock_price_df_copy['Date'] = pd.to_datetime(stock_price_df_copy['Date']).dt.normalize()
    news_df_copy['Date'] = pd.to_datetime(news_df_copy['Date']).dt.normalize()

    stock_price_df_copy.set_index('Date', inplace=True)
    news_df_copy.set_index('Date', inplace=True)

    # 按照日期进行排序
    stock_price_df_copy.sort_index(inplace=True)
    news_df_copy.sort_index(inplace=True)
    if sentiment_key_name == "Sentiment_gpt":
        # 将大于 5 的值替换为 5
        news_df_copy.loc[news_df_copy['Sentiment_gpt'] > 5, 'Sentiment_gpt'] = 5

        # 将小于 1 的值替换为 1
        news_df_copy.loc[news_df_copy['Sentiment_gpt'] < 1, 'Sentiment_gpt'] = 1

    # 3. 计算news.csv中每天的情感平均值
    average_sentiment = news_df_copy.groupby('Date')[sentiment_key_name].mean().reset_index()

    # 使用log递减规则填充缺失的日期
    average_sentiment_filled = fill_missing_dates_with_exponential_decay(average_sentiment, 'Date', sentiment_key_name, sentiment_key_name)

    # 4. 合并数据
    merged_df = pd.merge(stock_price_df_copy, average_sentiment_filled, on='Date', how='left')
    # 将NaN替换为3
    merged_df[sentiment_key_name].fillna(3, inplace=True)

    df_cleaned = merged_df.dropna(subset=['News_flag'])

    # 过滤掉sentiment列为0的行
    df_cleaned = df_cleaned[df_cleaned[sentiment_key_name] != 0]
    if sentiment_key_name == "Sentiment_gpt":
        df_cleaned['Scaled_sentiment'] = df_cleaned[sentiment_key_name].apply(lambda x: (x - 0.9999) / 4)
    elif sentiment_key_name == "Sentiment_blob":
        df_cleaned['Scaled_sentiment'] = df_cleaned[sentiment_key_name].apply(lambda x: (x + 1) / 2)
    # 打印合并后的数据框
    df_cleaned.columns.str.capitalize()
    # print(df_cleaned)
    print(len(df_cleaned['Close']))
    if len(df_cleaned['Close']) < 333:
        print(stock_price_csv_file)
        print("Lower than 333")
        return 0, df_cleaned
    return 1, df_cleaned


def start_inte(stock_price_folder_path, news_folder_path, saving_path, sentiment_key_name):
    # 1. 读取CSV文件并转换列名
    stock_price_csv_files = [file for file in os.listdir(stock_price_folder_path) if file.endswith('.csv')]
    for stock_price_csv_file in stock_price_csv_files:
        print(stock_price_csv_file)
        stock_file_path = os.path.join(stock_price_folder_path, stock_price_csv_file)
        stock_price_df = pd.read_csv(stock_file_path)
        stock_price_df.columns = stock_price_df.columns.str.capitalize()
        news_file_path = os.path.join(news_folder_path, stock_price_csv_file)
        if not os.path.isfile(news_file_path):
            print("No file storing corresponding stock news")
            continue
        news_df = pd.read_csv(news_file_path)
        news_df.columns = news_df.columns.str.capitalize()
        flag_333, merged_data = integrate_data(stock_price_df, news_df, stock_price_csv_file, sentiment_key_name)
        merged_data.to_csv(os.path.join(saving_path, stock_price_csv_file), index=False)


if __name__ == "__main__":
    stock_price_folder = "stock_price_data_preprocessed"
    news_folder = "news_data_sentiment_scored_by_gpt"
    saving_path = "gpt_sentiment_price_news_integrate"

    Sentiment_key_name = 'Sentiment_gpt'

    start_inte(stock_price_folder, news_folder, saving_path, Sentiment_key_name)

