# Dataset experiments
In this section, we provide code that can reproduce our experiments, , and for each model we have used a separate folder to store the experiments, e.g. folder `CNN-for-Time-Series-Prediction`
## 1. Test dataset on classic models: CNN, RNN, GRU, LSTM, and cutting-edge models：Transformer, TimesNet
In `dataset_test` folder, run `run.py` in the folder corresponding to each model
e.g. 
```
cd path/dataset_test/CNN-for-Time-Series-Prediction
python run.py
```

Results and trained models will be stored at corresponding folders e.g. `CNN-for-Time-Series-Prediction/saved_models`, `CNN-for-Time-Series-Prediction/test_result_50`(50 means using 50 csvs to test)
## 2. Evaluate by comparing their results
### Running `integrate_result.py` in folder `dataset_test` 
This creates the `integrated_eval_data_{n}` folder (n represents number of csvs used in test) and folder `integrated_predicted_data_{n}`.

The `integrated_eval_data_{n}` folder stores mse, mae and r2 information, and the `integrated_predicted_data_{n}` folder stores the predicted result data.

### Running `eval_output.py` in folder `dataset_test` 
This creates a table of comparisons based on the `integrated_eval_data_{n}` folder created before e.g. `merged_eval_data_5.csv`

## For example
Using CNN to test our dataset:

1. run `CNN-for-Time-Series-Prediction/run.py` to use our dataset to train CNN


| Folders             | Sub folders                                                | 
|---------------------|------------------------------------------------------------|
| test_result_5       | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
| test_result_25      | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
| test_result_50      | AMD_nonsentiment_2024013123, AMD_sentiment_2024013123, ... |
   
   you may get result like this

2. run `integrate_result.py` to integrate data

| Metric          | GRU           | CNN           | LSTM          | RNN           | TimesNet      | Transformer   |
|-----------------|---------------|---------------|---------------|---------------|---------------|---------------|
| MAE             | 0.013692475   | 0.024043111   | 0.014487029   | 0.030090562   | 0.028123611   | 0.009098691   |
| MSE             | 0.000354704   | 0.000897477   | 0.000406081   | 0.001951202   | 0.001378664   | 0.000118825   |
| R2              | 0.920132321   | 0.797917934   | 0.908563921   | 0.560653936   | 0.679685415   | 0.972400232   |
| Stock_symbol    | KO            | KO            | KO            | KO            | KO            | KO            |

   you may get integrated result like this
3. run `eval_output.py` to get `merged_eval_data_{n}.csv` to compare results from diffrent models

| Stock_symbol | If_sentiment  | GRU_MAE    | GRU_MSE    | GRU_R2     | CNN_MAE    | CNN_MSE    | CNN_R2     | LSTM_MAE   | LSTM_MSE   | LSTM_R2    | RNN_MAE    | RNN_MSE    | RNN_R2     | Transformer_MAE | Transformer_MSE | Transformer_R2 | TimesNet_MAE | TimesNet_MSE | TimesNet_R2 |
|--------------|---------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-----------------|-----------------|---------------|--------------|--------------|-------------|
| AMD          | sentiment     | 0.05606504 | 0.005572476| 0.811967523| 0.066068177| 0.007342226| 0.752250739| 0.048591717| 0.004481877| 0.848767681| 0.074345906| 0.009377636| 0.683569749| 0.004626686     | 3.04E-05        | 0.996386536    | 0.02066326   | 0.000793612  | 0.910305642 |
| AMD          | nonsentiment  | 0.047851457| 0.004417478| 0.850940726| 0.069047148| 0.00792996 | 0.732418767| 0.044213154| 0.003590023| 0.878861599| 0.052047684| 0.004885934| 0.835133567| 0.004859196     | 4.00E-05        | 0.995245374    | 0.021087526  | 0.000814597  | 0.907933901 |
| WMT          | sentiment     | 0.014154337| 0.000368689| 0.742596605| 0.017838694| 0.00053024 | 0.629808957| 0.012901138| 0.000297176| 0.792524285| 0.017634143| 0.000672053| 0.530801319| 0.003706584     | 3.53E-05        | 0.988483739    | 0.027463866  | 0.001175382  | 0.587828065 |
| WMT          | nonsentiment  | 0.012961662| 0.000308344| 0.784727299| 0.018902796| 0.000586256| 0.590701071| 0.012767703| 0.000301705| 0.789362262| 0.013839398| 0.00033947 | 0.762996591| 0.006190383     | 7.31E-05        | 0.976164151    | 0.02851286   | 0.001305128  | 0.542329827 |
| GOOG         | sentiment     | 0.023192957| 0.001044137| 0.772305224| 0.040671396| 0.002402573| 0.476071245| 0.024005472| 0.001115869| 0.756662669| 0.065988089| 0.006097407| -0.329660902| 0.002247489     | 7.70E-06        | 0.601947385    | 0.001815549  | 6.53E-06     | 0.656571803 |
| GOOG         | nonsentiment  | 0.022036099| 0.000988054| 0.784535107| 0.049649067| 0.003562549| 0.223115327| 0.021355772| 0.000938566| 0.795327041| 0.064505283| 0.00557126 | -0.214923915| 0.001564737     | 3.60E-06        | 0.813988402    | 0.002226247  | 8.18E-06     | 0.569788374 |
| TSM          | sentiment     | 0.030024087| 0.001606033| 0.930259946| 0.048718401| 0.003884207| 0.831332945| 0.029974532| 0.001593149| 0.930819415| 0.03721305 | 0.002513136| 0.890870092| 0.006924621     | 5.36E-05        | 0.992943875    | 0.016805931  | 0.00050976   | 0.930676544 |
| TSM          | nonsentiment  | 0.029591864| 0.001531021| 0.933517247| 0.053306702| 0.004308412| 0.812912373| 0.029814158| 0.001565802| 0.932006904| 0.041323378| 0.003055519| 0.867317726| 0.007152642     | 6.25E-05        | 0.991767259    | 0.019357385  | 0.000656321  | 0.910745388 |
| KO           | sentiment     | 0.012843123| 0.000317785| 0.928445322| 0.022584229| 0.000805671| 0.818589683| 0.01329065 | 0.000327492| 0.926259777| 0.032634155| 0.001607865| 0.637962204| 0.004542233     | 4.99E-05        | 0.988414696    | 0.029179125  | 0.001220426  | 0.716450061 |
| KO           | nonsentiment  | 0.014087051| 0.000383471| 0.913654938| 0.023775534| 0.000898275| 0.797738354| 0.013605789| 0.000358441| 0.919290925| 0.026498671| 0.001421359| 0.679957191| 0.006391796     | 8.39E-05        | 0.980501042    | 0.018263845  | 0.000594417  | 0.861894914 |
