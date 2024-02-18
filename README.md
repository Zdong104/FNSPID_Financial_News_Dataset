# FNSPID
In this github repo, we did three main tasks：
## 1. Data scraper. 
In folder `data_scraper`, we provided tools to collect news data from nasdaq.
## 2. Data processor.
In folder `data_processor`, we explained how we integrate our data into into workable data.
## 3. Dataset experiments.
In folder `dataset_test`, we provided ways using DL models to test dataset.
# Data scraper
`data_scraper` is aiming to collect news and price data of stocks
`data_scraper` is divided into 3 main steps:
1. use headline scraper to download the headline and url of the news
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

