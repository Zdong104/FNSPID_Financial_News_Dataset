import pandas as pd
import os
import glob


def re_syn(true_data, predicted_data):
    # 选取每个窗口的最后一个数据点
    true_data_processed = true_data.tolist()[:2] + true_data[2::3].tolist()

    # 计算Pm值
    predicted_data_processed = []
    n = len(predicted_data)
    m = ((n - 6) // 3) + 4
    # print(n, m)
    for i in range(1, m + 1):
        if i == 1:
            predicted_data_processed.append(predicted_data.iloc[0])
        elif i == 2:
            predicted_data_processed.append((predicted_data.iloc[1] + predicted_data.iloc[3]) / 2)
        elif i == m - 1:
            predicted_data_processed.append((predicted_data.iloc[-2] + predicted_data.iloc[-4]) / 2)
        elif i == m:
            predicted_data_processed.append(predicted_data.iloc[-1])
        else:
            idx = 3 * (i - 2) - 1
            predicted_data_processed.append(
                (predicted_data.iloc[idx] + predicted_data.iloc[idx + 2] + predicted_data.iloc[idx + 4]) / 3)
    result_df = pd.DataFrame(
        {'True_Data_origin': true_data_processed, 'Predicted_Data_origin': predicted_data_processed})
    # 创建一个只包含0的新行
    new_row = pd.DataFrame({'True_Data_origin': [None], 'Predicted_Data_origin': [None]})  # 其他列可以填充None或适当的默认值

    # 合并新行与原始数据
    df_updated = pd.concat([new_row, result_df], ignore_index=True)
    return df_updated


def integrate_Predicted_data(symbol, sentiment_type, n):
    prediction_name = symbol + "_" + sentiment_type
    captal_sentiment = symbol + "_Sentiment"
    captal_nonsentiment = symbol + "_Nonsentiment"
    model_dirs = [
        f"GRU-for-Time-Series-Prediction/test_result_{n}",
        f"CNN-for-Time-Series-Prediction/test_result_{n}",
        f"LSTM-for-Time-Series-Prediction/test_result_{n}",
        f"RNN-for-Time-Series-Prediction/test_result_{n}",
        f"TimesNet-for-Time-Series-Prediction/test_result_{n}",
        f"Transformer-for-Time-Series-Prediction/test_result_{n}",
    ]

    # 准备收集各个模型的DataFrame
    dataframes = []

    for model_dir in model_dirs:
        # 在每个模型的目录中查找包含指定符号的文件夹
        for folder in os.listdir(model_dir):
            if prediction_name in folder or captal_sentiment in folder or captal_nonsentiment in folder:
                full_folder_path = os.path.join(model_dir, folder)
                # print(full_folder_path)
                # 在找到的文件夹内寻找包含指定符号的CSV文件
                pattern = f"{full_folder_path}/*{symbol}*_predicted_data.csv"
                for file in glob.glob(pattern):
                    # print(file)
                    model_name = file.split('-')[0]
                    print(model_name)
                    df = pd.read_csv(file)
                    df_selected = df[["True_Data_origin", "Predicted_Data_origin"]]
                    for col in df.columns:
                        if col == 'truth':
                            df.rename(columns={col: 'True_Data_origin'}, inplace=True)
                        if col == 'predicted':
                            df.rename(columns={col: 'Predicted_Data_origin'}, inplace=True)
                    if model_name == "TimesNet":
                        df_selected = re_syn(df["True_Data_origin"], df["Predicted_Data_origin"])
                    df_selected = df_selected.rename(
                        columns={"Predicted_Data_origin": f"{model_name}_Predicted_Data_origin"})
                    # print(df_selected)
                    dataframes.append(df_selected)

    # 合并所有模型的预测数据到一个DataFrame
    merged_df = pd.concat([df.set_index(df.index) for df in dataframes], axis=1)
    # print(merged_df)
    # 删除重复的True_data列，保留第一个
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # 保存到新的CSV文件
    pred_directory = f'integrated_predicted_data_{n}'
    os.makedirs(pred_directory, exist_ok=True)
    merged_df.to_csv(f"integrated_predicted_data_{n}/{symbol}_{sentiment_type}_integrated_predicted_data.csv",
                     index=False)


def integrate_eval_data(symbol, sentiment_type, n):
    prediction_name = symbol + "_" + sentiment_type
    captal_sentiment = symbol + "_Sentiment"
    captal_nonsentiment = symbol + "_Nonsentiment"
    model_dirs = [
        f"GRU-for-Time-Series-Prediction/test_result_{n}",
        f"CNN-for-Time-Series-Prediction/test_result_{n}",
        f"LSTM-for-Time-Series-Prediction/test_result_{n}",
        f"RNN-for-Time-Series-Prediction/test_result_{n}",
        f"TimesNet-for-Time-Series-Prediction/test_result_{n}",
        f"Transformer-for-Time-Series-Prediction/test_result_{n}",
    ]

    # 准备收集各个模型的DataFrame
    dataframes = []

    for model_dir in model_dirs:
        # 在每个模型的目录中查找包含指定符号的文件夹
        for folder in os.listdir(model_dir):
            if prediction_name in folder or captal_sentiment in folder or captal_nonsentiment in folder:
                full_folder_path = os.path.join(model_dir, folder)
                # print(full_folder_path)
                # 在找到的文件夹内寻找包含指定符号的CSV文件
                pattern = f"{full_folder_path}/*{symbol}*_*eval*.csv"
                for file in glob.glob(pattern):
                    print(file)
                    model_name = file.split('-')[0]
                    # print(model_name)
                    df = pd.read_csv(file)
                    df = df.rename(
                        columns={"MAE": f"{model_name}_MAE", "MSE": f"{model_name}_MSE", "R2": f"{model_name}_R2"})
                    # print(df_selected)
                    dataframes.append(df)

    # 合并所有模型的预测数据到一个DataFrame
    merged_df = pd.concat([df.set_index(df.index) for df in dataframes], axis=1)
    # print(merged_df)
    # 删除重复的True_data列，保留第一个
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df['Stock_symbol'] = f'{symbol}'
    # 保存到新的CSV文件
    eval_directory = f'integrated_eval_data_{n}'
    os.makedirs(eval_directory, exist_ok=True)
    merged_df.to_csv(f"integrated_eval_data_{n}/{symbol}_{sentiment_type}_integrated_eval.csv", index=False)


if __name__ == "__main__":
    # 指定符号和模型目录
    symbols = ["AMD", "WMT", "KO", "TSM", "GOOG"]
    sentiment_types = ["sentiment", "nonsentiment"]
    nums_csvs = [5, 25, 50]
    for num_csvs in nums_csvs:
        n = num_csvs
        for symbol in symbols:
            for sentiment_type in sentiment_types:
                integrate_Predicted_data(symbol, sentiment_type, n)
                integrate_eval_data(symbol, sentiment_type, n)
