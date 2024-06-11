import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tst import Transformer
from tqdm import tqdm
import glob
import pdb



# Function to create sequences
# input_length = 50 
# output_length = 3
def create_sequences(data, input_length, output_length):
    X, y = [], []
    for i in range(0,(len(data) - input_length - output_length + 1), output_length):
        X.append(data[i:(i + input_length)]) # X shape should be [N,50,6]
        y.append(data[(i + input_length):(i + input_length + output_length), :]) # y shape should be [N,3,6]
    return np.array(X), np.array(y)






def read_csv_case_insensitive(file_path):
    try:
        # Convert the filename pattern to a case-insensitive glob pattern
        pattern = ''.join(['[{}{}]'.format(char.lower(), char.upper()) if char.isalpha() else char for char in file_path])
        
        # Use glob to find matching files
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Assuming you want to read the first matching file
            return pd.read_csv(matching_files[0])
        else:
            # If no files match, inform the user and return None or handle it as needed
            print(f"No file matches the pattern: {file_path}")
            return None
    except Exception as e:
        # If an error occurs, print the error message and return None or handle it as needed
        print(f"An error occurred: {e}")
        return None




def data_processor(data):
  # Checking if GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print('device = ',device)
  # print(data)
  # Scaling the data
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)

  # Creating sequences
  input_length = 50
  output_length = 3

  # Split training data into training and validation sets
  split_ratio = 0.85
  split = int(split_ratio * len(scaled_data))
  data_train = scaled_data[:split]
  data_test = scaled_data[split:]
  print('data_train',data_train.shape)
  print('data_test',data_test.shape)

  # Splitting the dataset into training and testing sets (80-20 split)
  X_train, y_train = create_sequences(data_train, input_length, output_length)
  X_test, y_test = create_sequences(data_test, input_length, output_length)

  # Displaying the shapes of the datasets to ensure correctness
  print('X_train: ',X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test',y_test.shape)
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  # Transposing to match model's input shape
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  # Create a DataLoader for training data
  batch_size = 64  # Adjust the batch size as needed
  dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return dataloader_train, dataloader_test, scaler


def train_model(dataloader_train, pred_flag, symbol ,num_csvs, d_input):

  """
  Parameters
  ----------
  d_input:
      Model input dimension.
  d_model:
      Dimension of the input vector.
  d_output:
      Model output dimension.
  q:
      Dimension of queries and keys.
  v:
      Dimension of values.
  h:
      Number of heads.
  N:
      Number of encoder and decoder layers to stack.
  attention_size:
      Number of backward elements to apply attention.
      Deactivated if ``None``. Default is ``None``.
  dropout:
      Dropout probability after each MHA or PFF block.
      Default is ``0.3``.
  chunk_mode:
      Switch between different MultiHeadAttention blocks.
      One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
  pe:
      Type of positional encoding to add.
      Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
  pe_period:
      If using the ``'regular'` pe, then we can define the period. Default is
      ``None``.
  """
    # Model parameters
  d_output = d_input # prediction length be 6, this is confirmed
  d_model = 32 # Lattent dim
  q = 8 # Query size
  v = 8 # Value size
  h = 8 # Number of heads
  N = 4 # Number of encoder and decoder to stack
  attention_size = 50 # Attention window size 这个和形状没有关系
  dropout = 0.1 # Dropout rate
  pe = 'regular' # Positional encoding
  chunk_mode = None
  output_length = 3
  # Creating sequences

  # Creating the model
  model = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
  # model = TimeSeriesTransformer(num_features, num_outputs, dim_val, n_heads, n_decoder_layers, dropout_rate).to(device)
  # print(model)

  model_path = f'model_saved/_{num_csvs}_{N}layers.pt'

  # initialize the epoch as 0, to prevent previous assigned value
  epochs = 0  # Adjust as needed
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path} onto {'CUDA' if torch.cuda.is_available() else 'CPU'}")

  if pred_flag: # general training 
    epochs = 100
  else: # particular training
    epochs = 50



  # Loss function and optimizer
  loss_function = nn.MSELoss()
  # optimizer = optim.Adam(model.parameters(), lr=0.001)
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  model.train()
  # Prepare loss history
  hist_loss = np.zeros(epochs)
  for idx_epoch in range(epochs):
      running_loss = 0
      # use fancy training percentage bar
      with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{epochs}]") as pbar:
          for idx_batch, (x, y) in enumerate(dataloader_train):
              optimizer.zero_grad()
              
              # Propagate input
              y_pred = model(x.to(device)) # torch.Size([64, 50, 6])

              y_pred_reshaped = y_pred[:, -int(output_length):, :]  # Take the last 3 slices along the second dimension, 本来是50个element的， 解码的时候我们只关心最后的三个

              # Comupte loss
              loss = loss_function(y.to(device), y_pred_reshaped) # [64,3,6] & [64,3,6] 
              # y shape is torch.Size([64, 50, 6])
              # Backpropage loss
              loss.backward()

              # Update weights
              optimizer.step()

              running_loss += loss.item()
              pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
              pbar.update(x.shape[0])
          
          train_loss = running_loss/len(dataloader_train)
          pbar.set_postfix({'loss': train_loss})
          
          hist_loss[idx_epoch] = train_loss
          
  print("Training complete.")
          
  plt.plot(hist_loss, 'o-', label='train')
  plt.legend()  
  plt.savefig(os.path.join("plot_saved", f"{symbol}_{num_stocks}_training_curve.pdf"))
  # Save the model
  os.makedirs(os.path.join("model_saved"), exist_ok=True)
  torch.save(model.state_dict(), model_path)
  print(f"saved model into {model_path}")

  return model



def eval_model(model, dataloader_test, symbol, num_csvs, scaler, output_length):
      # Prediction on test data
      predictions = []
      actuals = []
      model.eval()
      with torch.no_grad(): 
        # for x, y in enumerate(dataloader_test):
        for x, y in dataloader_test:
          modelout_pre  = model(x.to(device))
          modelout = modelout_pre[:, -int(output_length):, :]
          # print('modelout',modelout.shape)
          predictions.append(modelout.cpu().numpy())
          actuals.append(y.cpu().numpy())
      


      output_length = 3
      predictions_np = np.concatenate(predictions, axis=0) # (155, 3, 6)
      actuals_np = np.concatenate(actuals, axis=0) # (155, 3, 6)
      
      
      y_pred_reshaped = predictions_np.reshape(-1, 6)
      y_test_reshaped = actuals_np.reshape(-1, 6)
      print('y_pred_reshaped', y_pred_reshaped.shape)
      print('y_test_reshaped', y_test_reshaped.shape)


      y_test_origin = scaler.inverse_transform(y_test_reshaped)
      y_pred_origin = scaler.inverse_transform(y_pred_reshaped)


      print('y_test_origin', y_test_origin.shape)
      print('y_pred_origin', y_pred_origin.shape)


      # Create the directory for saving plots if it doesn't exist
      os.makedirs("plot_saved", exist_ok=True)

      # Plotting the results
      plt.figure(figsize=(10, 6))
      plt.plot(y_test_origin[:,2], label="Ground Truth", color='blue')  # Use the third element as the one for plotting 
      plt.plot(y_pred_origin[:,2], label="Predicted", color='red')  # Use the third element as the one for plotting 
      plt.title(f"{symbol}: Ground Truth vs Predicted")
      plt.xlabel("Time Steps")
      plt.ylabel("Values")
      plt.legend()

      # Save the plot as a PDF in the 'plot_saved' folder
      plt.savefig(os.path.join("plot_saved", f"{symbol}_{num_stocks}.pdf"))



      # Calculate metrics
      mse = mean_squared_error(y_test_origin, y_pred_origin) # 如果不指定， 就是一个overall 的 MSE， MAE，R^2
      mae = mean_absolute_error(y_test_origin, y_pred_origin)
      r2 = r2_score(y_test_origin, y_pred_origin)
      print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")
      eval_df = pd.DataFrame({
            'MAE': [mae],
            'MSE': [mse],
            'R2': [r2]
        })
      

      # print(y_test_origin)
      # print("___")
      # print(y_pred_origin)
      # Save the results to a CSV file
      date_str = datetime.now().strftime("%Y%m%d%H%M")
      # Assuming y_test_reshaped, y_pred_reshaped, y_test_origin, and y_pred_origin are arrays of the same shape

      predicted_data_results = pd.DataFrame({
          'True_Data_Volume': y_test_reshaped[0],
          'Predicted_Data_Volume': y_pred_reshaped[0],
          'True_Data_origin_Volume': y_test_origin[0],
          'Predicted_Data_origin_Volume': y_pred_origin[0],
          'True_Data_Open': y_test_reshaped[1],  # Assuming this is correct
          'Predicted_Data_Open': y_pred_reshaped[1],  # Assuming this is correct
          'True_Data_origin_Open': y_test_origin[1],  # Assuming this is correct
          'Predicted_Data_origin_Open': y_pred_origin[1],  # Assuming this is correct
          'True_Data_High': y_test_reshaped[2],  # Assuming this is correct
          'Predicted_Data_High': y_pred_reshaped[2],  # Assuming this is correct
          'True_Data_origin_High': y_test_origin[2],  # Assuming this is correct
          'Predicted_Data_origin_High': y_pred_origin[2],  # Assuming this is correct
          'True_Data_Low': y_test_reshaped[3],  # Assuming this is correct
          'Predicted_Data_Low': y_pred_reshaped[3],  # Assuming this is correct
          'True_Data_origin_Low': y_test_origin[3],  # Assuming this is correct
          'Predicted_Data_origin_Low': y_pred_origin[3],  # Assuming this is correct
          'True_Data_Close': y_test_reshaped[4],  # Assuming this is correct
          'Predicted_Data_Close': y_pred_reshaped[4],  # Assuming this is correct
          'True_Data_origin_Close': y_test_origin[4],  # Assuming this is correct
          'Predicted_Data_origin_Close': y_pred_origin[4],  # Assuming this is correct
          'True_Data_Scaled_sentiment': y_test_reshaped[5],  # Assuming this is correct
          'Predicted_Data_Scaled_sentiment': y_pred_reshaped[5],  # Assuming this is correct
          'True_Data_origin_Scaled_sentiment': y_test_origin[5],  # Assuming this is correct
          'Predicted_Data_origin_Scaled_sentiment': y_pred_origin[5]  # Assuming this is correct
      })

      saving_folder = os.path.join(f"test_result_{num_csvs}",f"{symbol}_{date_str}")
      os.makedirs(saving_folder, exist_ok=True)
      predicted_data_results_save_path = os.path.join(saving_folder, f'{symbol}_{date_str}_predicted_data.csv')
      predicted_data_results.to_csv(predicted_data_results_save_path, index=False)

      os.makedirs(saving_folder, exist_ok=True)
      eval_df_save_path = os.path.join(saving_folder, f'{symbol}_{date_str}_eval_data.csv')
      eval_df.to_csv(eval_df_save_path, index=False)
      print(f"saved predictions and evals to {predicted_data_results_save_path} and {eval_df_save_path}")
      return
   


def sentiment_predict(csv_data,symbol, num_csvs, pred_flag, pred_names):
  # Select relevant columns and calculate d_input before converting to NumPy array
  selected_columns = ['Volume', 'Open', 'High', 'Low', 'Close', 'Scaled_sentiment']
  data = csv_data[selected_columns].values
  d_input = len(selected_columns)  # Number of input features, should be 6   [N, M, 6] 其中的的6

  dataloader_train, dataloader_test, scaler = data_processor(data)
  model = train_model(dataloader_train, pred_flag, symbol ,num_csvs, d_input)

  if pred_flag:
    if symbol in pred_names:
      eval_model(model, dataloader_test, symbol, num_csvs, scaler, output_length=3)





# Test of 5 
names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv','WMT.csv']
# Test of 25 
names_25 = [
  #  'AAPL.csv', 'ABBV.csv','BABA.csv', 'BRK-B.csv',
            'bhp.csv', 'C.csv', 'COST.csv', 'CVX.csv','DIS.csv', 'GE.csv',
         'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv','QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv', 'WFC.csv', 'gsk.csv',
         'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv'] # 'AMZN.csv'
# Tes of 50
names_50 = [
   'aal.csv', 'AAPL.csv', 'ABBV.csv', 'amgn.csv','BABA.csv', 'bhp.csv',
    'biib.csv', 'bidu.csv', 'BRK-B.csv','C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv', 'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 
              'DIS.csv', 'ebay.csv','GE.csv','gild.csv', 'gld.csv', 'gsk.csv', 'INTC.csv', 
              'mrk.csv', 'MSFT.csv', 'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv',
                'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'uso.csv', 'v.csv',
                  'WFC.csv', 'xlf.csv','KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv', ] #'AMZN.csv' 'dal.csv',



# names_1 = ['TSM.csv']
# names_1 = ['fakedata.csv',]
# pred_names = ['TSM']
pred_names = ['KO','AMD',"TSM","GOOG",'WMT']

names = names_50
# num_stocks = 1
# num_stocks = len(names)
# num_stocks = 50
num_stocks = len(names)



for i in range(1,2):
  if_pred = False
  if  i == 1:
      if_pred = True
  # for sentiment_type in sentiment_types:
  for name in names:
      # Checking if GPU is available
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print('device = ',device)

      csv_data = read_csv_case_insensitive(os.path.join("data", name))
      symbol_name = name.split('.')[0]
      print(symbol_name)
      sentiment_predict(csv_data, symbol_name, num_stocks, if_pred, pred_names)
print('Training Complete')