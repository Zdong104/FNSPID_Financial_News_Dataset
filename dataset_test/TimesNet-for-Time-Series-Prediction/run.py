import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Function to create sequences
def create_sequences(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:(i + input_length)])
        y.append(data[(i + input_length):(i + input_length + output_length),
                 2])  # 2 is the index of 'Close' in input_features
    return np.array(X), np.array(y)


# Building the TimesNet Model with corrected architecture
class TimesNet(nn.Module):
    def __init__(self, input_features, sequence_length, output_length, num_layers=4):
        super(TimesNet, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_features if i == 0 else 64
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * sequence_length, output_length)

    def forward(self, x):
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        x = self.flatten(x)
        x = self.dense(x)
        return x


# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = ', device)


def sentiment_predict(csv_data, symbol, num_csvs, pred_flag):
    # Selecting specified columns and normalizing the data
    input_features = ['Volume', 'Open', 'Close', 'Scaled_sentiment']
    csv_data_selected = csv_data[input_features]

    input_length = 50
    output_length = 3
    split_ratio = 0.85
    split_idx = int(split_ratio * len(csv_data_selected))
    raw_train = csv_data_selected[:split_idx]
    raw_test  = csv_data_selected[split_idx:]

    # 2. Scale based on train only
    scaler = MinMaxScaler().fit(raw_train)
    train_scaled = scaler.transform(raw_train)
    test_scaled  = scaler.transform(raw_test)

    # 3. Build sequences
    X_train, y_train = create_sequences(train_scaled, input_length, output_length)
    X_test,  y_test  = create_sequences(test_scaled,  input_length, output_length)


    # # Splitting the data into training and testing sets

    # X_train, X_test = X[:split], X[split:]
    # y_train, y_test = y[:split], y[split:]

    print('X_train: ', X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test', y_test.shape)
    # Instantiate the model with correct input dimensions
    model = TimesNet(4, 50, 3, num_layers=4).to(device)
    model_path = f'model_saved/sentiment_{num_csvs}.pt'

    epochs = 0  # Adjust as needed
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path} onto {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if pred_flag:
        epochs = 50
    else:
        epochs = 100

    # Defining the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert numpy arrays to PyTorch tensors and run training for 1 epoch to validate
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2).to(
        device)  # Transposing to match model's input shape
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    # print("Check")
    # print(X_test_tensor.shape,X_test_tensor[:10])
    # print(y_test_tensor.shape,y_test_tensor)

    batch_size = 64
    num_batches = int(len(X_train) / batch_size)

    # model, loss_function, optimizer

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for b in range(num_batches):
            start_index = b * batch_size
            end_index = start_index + batch_size
            x_batch = X_train_tensor[start_index:end_index].to(device)
            y_batch = y_train_tensor[start_index:end_index].to(device)

            # Forward pass
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches}")

    print("Training complete.")

    # Save the model
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs(os.path.join("model_saved"), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"saved model into {model_path}")

    if pred_flag:
        print("predicting:", pred_names)
        if symbol in pred_names:
            # Prediction on test data
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
                device)  # Assuming 'device' is already defined and used previously
            X_test_tensor = X_test_tensor.transpose(1, 2)  # Transpose to match model's input shape
            with torch.no_grad():
                y_pred = model(X_test_tensor)

            # Convert predictions to CPU and numpy array
            y_pred_np = y_pred.cpu().numpy()

            # Reshape y_pred_np to have the same shape as y_test for each prediction
            y_pred_reshaped = y_pred_np.reshape(-1,
                                                output_length)  # output_length is the number of predicted values (e.g., 3)

            # Flatten y_test and y_pred_reshaped for comparison
            y_test_flattened = y_test.flatten()
            y_pred_flattened = y_pred_reshaped.flatten()
            # print("check")
            # print(y_test_flattened)
            # print(y_pred_flattened)
            # Calculate metrics
            mse = mean_squared_error(y_test_flattened, y_pred_flattened)
            mae = mean_absolute_error(y_test_flattened, y_pred_flattened)
            r2 = r2_score(y_test_flattened, y_pred_flattened)
            print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")
            eval_df = pd.DataFrame({
                'MAE': [mae],
                'MSE': [mse],
                'R2': [r2]
            })

            # Create the directory for saving plots if it doesn't exist
            os.makedirs("plot_saved", exist_ok=True)

            # Plotting the results
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_flattened, label="Ground Truth", color='blue')  # Assuming y_test_flattened is defined
            plt.plot(y_pred_flattened, label="Predicted", color='red')  # Assuming y_pred_flattened is defined
            plt.title(f"{symbol} - Sentiment: Ground Truth vs Predicted")
            plt.xlabel("Time Steps")
            plt.ylabel("Values")
            plt.legend()

            # Save the plot as a PDF in the 'plot_saved' folder
            plt.savefig(os.path.join("plot_saved", f"{symbol}_sentiment_{num_stocks}.pdf"))

            # 创建一个形状为 [-1, 4] 的全零数组
            y_test_expanded = np.zeros((y_test_flattened.shape[0], 4))
            y_pred_expanded = np.zeros((y_pred_flattened.shape[0], 4))

            # 将原始数据放在第三列（索引为2）
            y_test_expanded[:, 2] = y_test_flattened
            y_pred_expanded[:, 2] = y_pred_flattened
            y_test_origin = scaler.inverse_transform(y_test_expanded)[:, 2]
            y_pred_origin = scaler.inverse_transform(y_pred_expanded)[:, 2]
            # print(y_test_origin)
            # print("___")
            # print(y_pred_origin)
            # Save the results to a CSV file
            predicted_data_results = pd.DataFrame(
                {'True_Data': y_test_flattened, 'Predicted_Data': y_pred_flattened, 'True_Data_origin': y_test_origin,
                 'Predicted_Data_origin': y_pred_origin}, )
            saving_folder = os.path.join(f"test_result_{num_csvs}", f"{symbol}_sentiment_{date_str}")
            os.makedirs(saving_folder, exist_ok=True)
            predicted_data_results_save_path = os.path.join(saving_folder,
                                                            f'{symbol}_sentiment_{date_str}_predicted_data.csv')
            predicted_data_results.to_csv(predicted_data_results_save_path, index=False)

            os.makedirs(saving_folder, exist_ok=True)
            eval_df_save_path = os.path.join(saving_folder, f'{symbol}_sentiment_{date_str}_eval_data.csv')
            eval_df.to_csv(eval_df_save_path, index=False)
            print(f"saved predictions and evals to {predicted_data_results_save_path} and {eval_df_save_path}")


def nonsentiment_predict(csv_data, symbol, num_csvs, pred_flag):
    # Selecting specified columns and normalizing the data
    input_features = ['Volume', 'Open', 'Close']
    csv_data_selected = csv_data[input_features]

    input_length = 50
    output_length = 3
    split_ratio = 0.85
    split_idx = int(split_ratio * len(csv_data_selected))
    raw_train = csv_data_selected[:split_idx]
    raw_test  = csv_data_selected[split_idx:]

    # 2. Scale based on train only
    scaler = MinMaxScaler().fit(raw_train)
    train_scaled = scaler.transform(raw_train)
    test_scaled  = scaler.transform(raw_test)

    # 3. Build sequences
    X_train, y_train = create_sequences(train_scaled, input_length, output_length)
    X_test,  y_test  = create_sequences(test_scaled,  input_length, output_length)

    # # Splitting the data into training and testing sets

    # X_train, X_test = X[:split], X[split:]
    # y_train, y_test = y[:split], y[split:]

    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    # Instantiate the model with correct input dimensions
    model = TimesNet(3, 50, 3, num_layers=4).to(device)
    model_path = f'model_saved/nonsentiment_{num_csvs}.pt'

    epochs = 0  # Adjust as needed
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path} onto {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if pred_flag:
        epochs = 50
    else:
        epochs = 100

    # model.load_state_dict(torch.load(model_path, map_location=device))

    # Defining the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert numpy arrays to PyTorch tensors and run training for 1 epoch to validate
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).transpose(1,
                                                                          2)  # Transposing to match model's input shape
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    batch_size = 64
    num_batches = int(len(X_train) / batch_size)

    model, loss_function, optimizer

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for b in range(num_batches):
            start_index = b * batch_size
            end_index = start_index + batch_size
            x_batch = X_train_tensor[start_index:end_index].to(device)
            y_batch = y_train_tensor[start_index:end_index].to(device)

            # Forward pass
            y_pred = model(x_batch)
            loss = loss_function(y_pred, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches}")

    print("Training complete.")

    # Save the model
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs(os.path.join("model_saved"), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"saved model into {model_path}")

    if pred_flag:
        print("predicting:", pred_names)
        if symbol in pred_names:
            # Prediction on test data
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
                device)  # Assuming 'device' is already defined and used previously
            X_test_tensor = X_test_tensor.transpose(1, 2)  # Transpose to match model's input shape

            with torch.no_grad():
                y_pred = model(X_test_tensor)

            # Convert predictions to CPU and numpy array
            y_pred_np = y_pred.cpu().numpy()

            # Reshape y_pred_np to have the same shape as y_test for each prediction
            y_pred_reshaped = y_pred_np.reshape(-1,
                                                output_length)  # output_length is the number of predicted values (e.g., 3)

            # Flatten y_test and y_pred_reshaped for comparison
            y_test_flattened = y_test.flatten()
            y_pred_flattened = y_pred_reshaped.flatten()
            # Calculate metrics
            mse = mean_squared_error(y_test_flattened, y_pred_flattened)
            mae = mean_absolute_error(y_test_flattened, y_pred_flattened)
            r2 = r2_score(y_test_flattened, y_pred_flattened)
            print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")
            eval_df = pd.DataFrame({
                'MAE': [mae],
                'MSE': [mse],
                'R2': [r2]
            })

            # Create the directory for saving plots if it doesn't exist
            os.makedirs("plot_saved", exist_ok=True)

            # Plotting the results
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_flattened, label="Ground Truth", color='blue')  # Assuming y_test_flattened is defined
            plt.plot(y_pred_flattened, label="Predicted", color='red')  # Assuming y_pred_flattened is defined
            plt.title(f"{symbol} - Nonsentiment: Ground Truth vs Predicted")
            plt.xlabel("Time Steps")
            plt.ylabel("Values")
            plt.legend()

            # Save the plot as a PDF in the 'plot_saved' folder
            plt.savefig(os.path.join("plot_saved", f"{symbol}_nonsentiment_{num_stocks}.pdf"))

            # 创建一个形状为 [-1, 3] 的全零数组
            y_test_expanded = np.zeros((y_test_flattened.shape[0], 3))
            y_pred_expanded = np.zeros((y_pred_flattened.shape[0], 3))

            # 将原始数据放在第三列（索引为2）
            y_test_expanded[:, 2] = y_test_flattened
            y_pred_expanded[:, 2] = y_pred_flattened
            y_test_origin = scaler.inverse_transform(y_test_expanded)[:, 2]
            y_pred_origin = scaler.inverse_transform(y_pred_expanded)[:, 2]
            # print(y_test_origin)
            # print("___")
            # print(y_pred_origin)
            # Save the results to a CSV file
            predicted_data_results = pd.DataFrame(
                {'True_Data': y_test_flattened, 'Predicted_Data': y_pred_flattened, 'True_Data_origin': y_test_origin,
                 'Predicted_Data_origin': y_pred_origin}, )
            saving_folder = os.path.join(f"test_result_{num_csvs}", f"{symbol}_nonsentiment_{date_str}")
            os.makedirs(saving_folder, exist_ok=True)
            predicted_data_results_save_path = os.path.join(saving_folder,
                                                            f'{symbol}_nonsentiment_{date_str}_predicted_data.csv')
            predicted_data_results.to_csv(predicted_data_results_save_path, index=False)

            os.makedirs(saving_folder, exist_ok=True)
            eval_df_save_path = os.path.join(saving_folder, f'{symbol}_nonsentiment_{date_str}_eval_data.csv')
            eval_df.to_csv(eval_df_save_path, index=False)
            print(f"saved predictions and evals to {predicted_data_results_save_path} and {eval_df_save_path}")


if __name__ == "__main__":
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
                for name in names:
                    print(name)
                    csv_data = pd.read_csv(os.path.join("data", name))
                    symbol_name = name.split('.')[0]
                    print(symbol_name)
                    sentiment_predict(csv_data, symbol_name, num_stocks, if_pred)
                    nonsentiment_predict(csv_data, symbol_name, num_stocks, if_pred)
