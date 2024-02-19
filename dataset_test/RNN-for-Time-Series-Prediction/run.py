__copyright__ = "Fan xinyu 2024"

import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from core.data_processor import DataLoader
from core.RNN_modified_model import Model

# 生成带有当前时间的文件夹名
current_time = datetime.now().strftime("%Y%m%d%H")


def output_results_and_errors_multiple(predicted_data, true_data, true_data_base, prediction_len, file_name,
                                       sentiment_type, num_csvs):
    ### 输出预测和真实值
    # 创建一个空的DataFrame
    save_df = pd.DataFrame()

    # 将真实值添加到DataFrame中
    save_df['True_Data'] = true_data.reshape(-1)
    save_df['Base'] = true_data_base.reshape(-1)

    # 转化回原scale
    save_df['True_Data_origin'] = (save_df['True_Data'] + 1) * save_df['Base']

    # 将所有预测数据拼接在一起
    if predicted_data:
        # 使用列表推导式将数组拼接
        all_predicted_data = np.concatenate([p for p in predicted_data])
    else:
        # 如果 predicted_data 为空，则赋值为一个空数组或者根据你的需求进行处理
        all_predicted_data = predicted_data

    file_name = file_name.split(".")[0]
    sentiment_type = str(sentiment_type)

    # 将拼接后的预测数据添加到DataFrame中
    save_df['Predicted_Data'] = pd.Series(all_predicted_data)

    # 转化回原scale
    save_df['Predicted_Data_origin'] = (save_df['Predicted_Data'] + 1) * save_df['Base']

    # 如果预测值的长度不同，则填充NaN
    save_df = save_df.fillna(np.nan)
    result_folder = f"test_result_{num_csvs}"
    save_file_path = os.path.join(result_folder, f"{file_name}_{sentiment_type}_{current_time}",
                                  f"{file_name}_{sentiment_type}_{current_time}_predicted_data.csv")
    # 保存DataFrame到CSV文件
    # 创建目录（如果不存在）
    os.makedirs(os.path.join(result_folder, f"{file_name}_{sentiment_type}_{current_time}"), exist_ok=True)

    save_df.to_csv(save_file_path, index=False)
    print(f"Data saved to {save_file_path}")
    ### 输出eval
    # 截断数据以确保长度一致
    min_length = min(len(save_df['Predicted_Data']), len(save_df['True_Data']))
    predicted_data = save_df['Predicted_Data'][:min_length]
    true_data = save_df['True_Data'][:min_length]

    # 计算 MAE, MSE, R²
    mae = mean_absolute_error(true_data, predicted_data)
    mse = mean_squared_error(true_data, predicted_data)
    r2 = r2_score(true_data, predicted_data)

    # print("MAE:", mae)
    # print("MSE:", mse)
    # print("R²:", r2)
    results_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2]
    })

    eval_file_path = os.path.join(result_folder, f"{file_name}_{sentiment_type}_{current_time}",
                                  f"{file_name}_{sentiment_type}_{current_time}_eval.csv")

    # 保存结果到CSV文件
    results_df.to_csv(eval_file_path, index=False)
    print(f"\nResults saved to {eval_file_path}")


# Main Function
def main(configs, data_filename, sentiment_type, flag_pred, model_name, num_csvs):
    symbol_name = name.split('.')[0]
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', data_filename),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['columns_to_normalise'],
        configs['data']['prediction_length']
    )

    model = Model()
    model_path = f"saved_models/{model_name}_{sentiment_type}_{num_csvs}.h5"
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model.build_model(configs)

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print("X:", x.shape)
    # print(x[0])
    print("Y:", y.shape)
    # print(y)
    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir'],
        sentiment_type=sentiment_type,
        model_name=model_name,
        num_csvs=num_csvs
    )
    if flag_pred:
        if symbol_name in pred_names:
            print("-----Predicting-----")
            x_test, y_test, y_base = data.get_test_data(
                seq_len=configs['data']['sequence_length'],
                normalise=configs['data']['normalise'],
                cols_to_norm=configs['data']['columns_to_normalise']
            )
            print("test data:")
            print("X:", x_test.shape)
            print("Y:", y_test.shape)
            predictions = model.predict_sequences_multiple_modified(x_test, configs['data']['sequence_length'],
                                                                    configs['data']['prediction_length'])

            output_results_and_errors_multiple(predictions, y_test, y_base, configs['data']['prediction_length'],
                                               symbol_name, sentiment_type, num_csvs)


if __name__ == '__main__':
    model_name = "RNN"
    sentiment_types = ["nonsentiment"]  # "sentiment",
    # Test csvs = 25
    # names = ['AAPL.csv', 'ABBV.csv', 'ACGLO.csv', 'AFGD.csv', 'AGM-A.csv', 'AKO-A.csv', 'AMD.csv', 'AMZN.csv', 'ARTLW.csv', 'BABA.csv', 'BCDAW.csv', 'BH-A.csv', 'BHFAL.csv', 'BRK-B.csv', 'BROGW.csv', 'C.csv', 'CIG-C.csv', 'CLSN.csv', 'COST.csv', 'CRD-A.csv', 'CVX.csv', 'DIS.csv', 'FDEV.csv', 'FITBO.csv', 'GAINL.csv', 'GE.csv', 'GECCM.csv', 'GOOG.csv', 'GRP-UN.csv', 'GTN-A.csv', 'HCXY.csv', 'HVT-A.csv', 'INBKZ.csv', 'INTC.csv', 'KO.csv', 'MSFT.csv', 'NVDA.csv', 'OCFCP.csv', 'PBR-A.csv', 'PYPL.csv', 'QQQ.csv', 'QVCD.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'TSM.csv', 'UCBIO.csv', 'WFC.csv', 'WMT.csv', 'WSO-B.csv']

    # Test csvs = 50
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
                'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
                'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
                'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
                'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
                'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
                'WMT.csv', 'xlf.csv']

    # Test csvs = 25
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv', 'DIS.csv',
                'GE.csv',
                'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv',
                'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    # Test csvs = 5
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    all_names = [names_5, names_25, names_50]
    pred_names = ['KO', 'AMD', "TSM", "GOOG", 'WMT']
    for names in all_names:
        num_stocks = len(names)
        # num_stocks = 5
        # num_stocks = 25
        # num_stocks = 50
        # For the first and second runs, only model training was performed
        # In the third run, it will train and make predictions
        for i in range(3):
            if_pred = False
            if i == 2:
                if_pred = True
            for sentiment_type in sentiment_types:
                for name in names:
                    print(name)
                    configs = json.load(open(sentiment_type + '_config.json', 'r'))
                    main(configs, name, sentiment_type, if_pred, model_name, num_stocks)
