import os
import glob
import pandas as pd

import pandas as pd
import os
import glob


def merge_csv_files(symbols, directory, n):
    merged_data = []

    for symbol in symbols:
        sentiment_pattern = os.path.join(directory, f"{symbol}*_sentiment_integrated_eval*.csv")
        nonsentiment_pattern = os.path.join(directory, f"{symbol}*_nonsentiment_integrated_eval*.csv")

        # Using glob to find files that match the patterns
        sentiment_files = glob.glob(sentiment_pattern)
        nonsentiment_files = glob.glob(nonsentiment_pattern)

        if sentiment_files and nonsentiment_files:
            print(sentiment_files)
            print(nonsentiment_files)
            # Assuming the first file is the desired one
            sentiment_df = pd.read_csv(sentiment_files[0])
            nonsentiment_df = pd.read_csv(nonsentiment_files[0])

            # Extracting and formatting data
            sentiment_data = sentiment_df[
                ['Stock_symbol',
                 "GRU_MAE", "GRU_MSE", 'GRU_R2', 
                 'CNN_MAE', 'CNN_MSE', 'CNN_R2', 
                 'LSTM_MAE', 'LSTM_MSE', 'LSTM_R2',
                 'RNN_MAE', 'RNN_MSE', 'RNN_R2',
                 'Transformer_MAE', "Transformer_MSE", "Transformer_R2",
                 'TimesNet_MAE', "TimesNet_MSE", "TimesNet_R2"]].iloc[0].tolist()
            sentiment_data.insert(1, 'sentiment')

            nonsentiment_data = nonsentiment_df[
                ['Stock_symbol',
                 "GRU_MAE", "GRU_MSE", 'GRU_R2',
                 'CNN_MAE', 'CNN_MSE', 'CNN_R2',
                 'LSTM_MAE', 'LSTM_MSE', 'LSTM_R2',
                 'RNN_MAE', 'RNN_MSE', 'RNN_R2',
                 'Transformer_MAE', "Transformer_MSE", "Transformer_R2",
                 'TimesNet_MAE', "TimesNet_MSE", "TimesNet_R2"]].iloc[0].tolist()
            nonsentiment_data.insert(1, 'nonsentiment')

            # Append to merged data
            merged_data.append(sentiment_data)
            merged_data.append(nonsentiment_data)

    # Creating a DataFrame from the merged data
    merged_df = pd.DataFrame(merged_data,
                             columns=['Stock_symbol', 'If_sentiment',
                                      "GRU_MAE", "GRU_MSE", 'GRU_R2', 'CNN_MAE',
                                      'CNN_MSE', 'CNN_R2',
                                        'LSTM_MAE', 'LSTM_MSE', 'LSTM_R2',
                                        'RNN_MAE', 'RNN_MSE', 'RNN_R2',
                                      'Transformer_MAE', "Transformer_MSE", "Transformer_R2",
                                      'TimesNet_MAE', "TimesNet_MSE", "TimesNet_R2"])
    print(merged_df)
    # Save to new CSV file
    merged_df.to_csv(f"merged_eval_data_{n}.csv", index=False)


if __name__ == "__main__":
    # 指定符号和模型目录
    nums_csvs = [5, 25, 50]
    for num_csvs in nums_csvs:
        n = num_csvs
        symbols = ["AMD", "WMT", "GOOG", "TSM", "KO"]  # Example symbols, replace with your actual list
        directory = f"integrated_eval_data_{n}"  # Replace with the actual directory path

        merge_csv_files(symbols, directory, n)
