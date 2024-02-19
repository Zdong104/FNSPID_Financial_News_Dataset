# FNSPID
** We appreciate your interest in stopping by our repo, This repo is still developing, the full version will be done before Feb 25, 2024. **
Meanwhile, we will include the tool, dataset and trained model and code. 


In this GitHub repo, we did three main tasks：
## 1. Data scraper. 
In the folder `data_scraper`, we provided tools to collect news data from Nasdaq.
## 2. Data processor.
In the folder `data_processor`, we explained how we integrate our data into workable data.
## 3. Dataset experiments.
In folder `dataset_test`, we provided ways using DL models to test the dataset.
# Data scraper
`data_scraper` aims to collect news and price data on stocks
`data_scraper` is divided into 3 main steps:
1. use headline scraper to download the headline and URL of the news
2. use the news content scraper to download the text of the news
3. use stock price scraper to download stock price

Before using it, please install `Selenium` and the corresponding browser driver, and then follow the `requirements.txt` to configure the environment

`pip install -r requirements.txt`

## 1. headline scraper：
### Before running the scraper, make sure that your current working directory is the `headline_scraper` folder

`cd path/data_scraper/headline_scraper`

 1. Put all the lists in `list_original` into the `headline_scraper/lists` folder

The `list_original` folder contains stock symbols starting with a-z in alphabetical order
 2. run `find_headlines v0.2.py`
 3. Type `a-z`, `a` for `list_a.csv` and `abc` for `list_a.csv`, `list_b.csv`, `list_c.csv`




![1707098657333](https://github.com/Zdong104/FNSPID/assets/91862936/9db14d61-9d44-4bcf-89d9-282de88238fd)

 4. Processed headlines will be stored in the `headline_scraper/headlines` folder according to the stock symbol, e.g. `aapl.csv`, `ibm.csv` ...

## 2. news content scraper:
### Before running the scraper, make sure that your current working directory is the `News_content_scraper` folder

`cd path/data_scraper/news_content_scraper`
1. Put all the lists in `list_original` into the `news_content_scraper/lists` folder
2. Put all files in `Headline_scraper/headlines` into `News_content_scraper/news_contents`
3. run 'add_Mark.py' first
4. run `find_content v0.2.py`
5. Type `a-z`, as mentioned above
6. Processed headlines will be stored in the `news_content_scraper/news_contents` folder

## 3. stock price scraper:
### Before running the scraper, make sure that your current working directory is the `stock_price_scraper` folder

`cd path/data_scraper/stock_price_scraper`
1. Put all the lists in `list_original` into the `stock_price_scraper/lists` folder
2. run 'add_Mark.py' first
4. run `get_price_from_yahoo.py`
5. Type `a-z`, as mentioned above
6. Processed headlines will be stored in the `stock_price_scraper/yahoo` folder

# Data processor
`data_processor` is used to score the collected news data with sentiment and then process it into a format that can be used to train model
`data_scraper` is divided into 4 main steps:
## 1. `preprocess.py`
This operation involves removing invalid and redundant price data and news data, and then converting the time to UTC format so that subsequent price data and news data can be aligned.
## 2. `summarize.py`
This operation uses Sumy library to summarise the news data with four algorithms, LSA, LexRank, Luhn and SumBasic, to obtain the summarised text
## 3. `score_by_gpt.py`
This operation selects one of the four summarised texts obtained in operation 2 and feed it to gpt according to our `prompt`, so that gpt can give a sentiment score for the given stock symbol
## 4. `price_news_integrate.py`
This operation integrates the news data with the price data by date and uses exponential decay for dates without news

# Dataset experiments
## 1. Test dataset on classic models: CNN, RNN, GRU, LSTM, and cutting-edge models：Transformer, TimesNet
In `dataset_test` folder, run `run.py` in the folder corresponding to each model, e.g. `CNN-for-Time-Series-Prediction/run.py`, `TimesNet-for-Time-Series-Prediction/run.py`

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




# Disclaimer
## Reliability and Security

The code provided in this GitHub repository is shared without any guarantee for its reliability and security. The developers and contributors of this project expressly disclaim any warranty, either implied or explicit, regarding the code's performance, security, or suitability for any particular purpose. The users should employ this code at their own risk, acknowledging that the developers shall not be held responsible for any damages or issues arising from its use.


## Purpose of Use

This code is primarily intended to illustrate our workflow processes and to serve as a medium for educational exchange and learning among users. It is made available for the purpose of showcasing our technical approaches and facilitating learning within the community. It is not designed for direct application in production environments or critical systems.

## Prohibition of Commercial Use

The use of this code for commercial purposes is strictly prohibited without prior authorization. If you wish to utilize this code in a commercial setting or for any revenue-generating activities, you are required to obtain explicit permission from the original authors. Please contact us at puma122707@gmail.com to discuss licensing arrangements or to seek approval for commercial use.


## Acknowledgement

By accessing, using, or contributing to this code, you acknowledge having read this disclaimer and agree to its terms. If you do not agree with these conditions, you should refrain from using or interacting with the code in any manner.



## Citation
```bibtex
@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series}, 
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}

