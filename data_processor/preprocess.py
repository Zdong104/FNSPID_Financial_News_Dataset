import os
from datetime import timedelta
from datetime import datetime
import pandas as pd


# 反推相对时间
def convert_to_utc(time_str):
    # 检查并去除时区缩写
    if " EDT" in time_str:
        time_str_cleaned = time_str.replace(" EDT", "")
        offset = timedelta(hours=-4)
    elif " EST" in time_str:
        time_str_cleaned = time_str.replace(" EST", "")
        offset = timedelta(hours=-5)
    else:
        # 默认为0时差，对于只有日期的情况不调整时区
        offset = timedelta(hours=0)
        time_str_cleaned = time_str

    # 尝试不同的日期时间格式
    formats = [
        '%B %d, %Y — %I:%M %p',  # "September 12, 2023 — 06:15 pm"
        '%b %d, %Y %I:%M%p',  # "Nov 14, 2023 7:35AM"
        '%d-%b-%y',  # "6-Jan-22"
        '%Y-%m-%d',  # "2021-4-5"
        '%Y/%m/%d',  # "2021/4/5"
        '%b %d, %Y'  # "DEC 7, 2023"
    ]

    for fmt in formats:
        try:
            # 尝试解析日期和时间
            dt = datetime.strptime(time_str_cleaned, fmt)
            # 如果格式只包含日期，不包含具体时间，则不应用时区调整
            if fmt == '%d-%b-%y':
                offset = timedelta(hours=0)

            # 调整为UTC时间
            dt_utc = dt + offset

            return dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            continue

    # 如果所有格式都不匹配，返回错误信息
    return "Invalid date format"


def date_inte(folder_path, saving_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for csv_file in csv_files:
        print('Starting: ' + csv_file)
        file_path = os.path.join(folder_path, csv_file)
        # 使用pandas的read_csv函数读取CSV文件
        df = pd.read_csv(file_path, on_bad_lines="warn")
        df.columns = df.columns.str.capitalize()
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        # 应用转换函数
        print(df["Date"])
        df['Date'] = df['Date'].apply(convert_to_utc)
        print(df["Date"])
        # # 将Date列转换为日期时间格式
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        # 按照Date列降序排序
        df = df.sort_values(by='Date', ascending=False)
        # 输出结果
        print(df)

        df.to_csv(os.path.join(saving_path, csv_file), index=False)
        print('Done: ' + csv_file)


if __name__ == "__main__":
    news_folder_path = 'news_data_raw'
    news_saving_path = 'news_data_preprocessed'

    stock_folder_path = 'stock_price_data_raw'
    stock_saving_path = 'stock_price_data_preprocessed'

    date_inte(news_folder_path, news_saving_path)
    date_inte(stock_folder_path, stock_saving_path)

