# Data processor
`data_processor` is used to score the collected news data with sentiment and then process it into a format that can be used to train model
`data_processor` is divided into 4 main steps:
### Before running the scraper, make sure that your current working directory is the `data_processor` folder
`cd path/data_processor`

## 1. Preprocess
Run `preprocess.py`

This operation involves removing invalid and redundant price data and news data, and then converting the time to UTC format so that subsequent price data and news data can be aligned.
## 2. Summarize news by sumy
Run `summarize.py`

This operation uses Sumy library to summarise the news data with four algorithms, LSA, LexRank, Luhn and SumBasic, to obtain the summarised text
## 3. Give sentiment scroe to summarized news text by GPT
Run`score_by_gpt.py`

This operation selects one of the four summarised texts obtained in operation 2 and feed it to gpt according to our `prompt`, so that gpt can give a sentiment score for the given stock symbol
## 4. Integrate price data and news data over time
Run `price_news_integrate.py`

This operation integrates the news data with the price data by date and uses exponential decay for dates without news
