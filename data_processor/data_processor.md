# Data processor
`data_processor` is used to score the collected news data with sentiment and then process it into a format that can be used to train model
`data_processor` is divided into 4 main steps:
### Before running the scraper, make sure that your current working directory is the `data_processor` folder
`cd path/data_processor`

## 1. Preprocess
Run `preprocess.py`

This operation involves removing invalid and redundant price data and news data, and then converting the time to UTC format so that subsequent price data and news data can be aligned.

e.g `aa.csv`

Before preprocess:`news_data_raw/aa.csv`
| Date                        | Url                                                                                                      | Text   | Mark |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|------|
| January 17, 2024 — 08:52 am EST | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | Long news text | 1    |

After preprocess:`news_data_preprocessed/aa.csv`

| Date                        | Url                                                                                                      | Text   | Mark |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|------|
| 2024-01-17 03:52:00+00:00 | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | Long news text | 1    |

## 2. Summarize news by sumy
Run `summarize.py`

This operation uses Sumy library to summarise the news data with four algorithms, LSA, LexRank, Luhn and SumBasic, to obtain the summarised text

e.g `aa.csv`

Before summarize:`news_data_preprocessed/aa.csv`
| Date                        | Url                                                                                                      | Text   | Mark |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|------|
| 2024-01-17 03:52:00+00:00 | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | March S&P 500 E-Mini futures (ESH24) are trending down -0.42% this morning as investors digested weak economic data from China while also gearing up for crucial U.S. retail sales data. In Tuesday’s trading session, Wall Street’s major averages closed in the red, (1689 more words) | 1    |

After summarize:`news_data_summarized/aa.csv`
| Date                        | Url                                                                                                      | Text   | Mark |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|------|
| 2024-01-17 03:52:00+00:00 | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | On the earnings front, notable companies like Prologis (PLD), Charles Schwab (SCHW), U.S. Bancorp (USB), Kinder Morgan (KMI), and Alcoa (AA) are set to report their quarterly figures today.        (2 more sentences) | 1    |


## 3. Give sentiment scroe to summarized news text by GPT
Run`score_by_gpt.py`

This operation selects one of the four summarised texts obtained in operation 2 and feed it to gpt according to our `prompt`, so that gpt can give a sentiment score for the given stock symbol

e.g `aa.csv`

Before score by gpt:`news_data_summarized/aa.csv`
| Date                        | Url                                                                                                      | Text   | Mark |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|------|
| 2024-01-17 03:52:00+00:00 | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | On the earnings front, notable companies like Prologis (PLD), Charles Schwab (SCHW), U.S. Bancorp (USB), Kinder Morgan (KMI), and Alcoa (AA) are set to report their quarterly figures today.        (2 more sentences) | 1    |

After score by gpt:`news_data_summarized/aa.csv`
| Date                        | Url                                                                                                      | Sentiment_gpt   |
|-----------------------------|----------------------------------------------------------------------------------------------------------|--------|
| 2024-01-17 03:52:00+00:00 | [https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints](https://www.nasdaq.com/articles/sp-futures-slip-ahead-of-key-u.s.-retail-sales-data-chinese-data-disappoints) | 3 |


## 4. Integrate price data and news data over time
Run `price_news_integrate.py`

This operation integrates the news data with the price data by date and uses exponential decay for dates without news

e.g `aa.csv`

preprocessed price data:`stock_price_data_preprocessed/aa.csv`
| Date                         | Open         | High         | Low          | Close       | Adj Close   | Volume  |
|------------------------------|--------------|--------------|--------------|-------------|-------------|---------|
| 2024-02-02 00:00:00+00:00 | 29           | 29.71999931  | 28.54999924  | 29.48999977 | 29.48999977 | 4954000 |
| 2024-02-01 00:00:00+00:00 | 30.07999992  | 30.40500069  | 29.14999962  | 29.69000053 | 29.69000053 | 4174600 |
| 2024-01-31 00:00:00+00:00 | 30.48999977  | 31.36000061  | 29.71500015  | 29.75       | 29.75       | 5760400 |
| 2024-01-30 00:00:00+00:00 | 30.34000015  | 30.84000015  | 30           | 30.61000061 | 30.61000061 | 4714700 |
....

integrated data:`gpt_sentiment_price_news_integrate/aa.csv`
| Date                       | Open        | High        | Low         | Close       | Adj Close   | Volume   | Sentiment_gpt | News_flag | Scaled_sentiment |
|----------------------------|-------------|-------------|-------------|-------------|-------------|----------|---------------|-----------|------------------|
| 2016-03-22 00:00:00+00:00 | 23.4532795  | 23.93387985 | 23.26103973 | 23.66954994 | 22.9438591  | 8258094  | 4             | 1         | 0.750025         |
| 2016-03-23 00:00:00+00:00 | 23.28507042 | 23.54940033 | 22.22775078 | 22.39595985 | 21.70932007 | 10581107 | 3.951229425   | 0         | 0.737832356      |
| 2016-03-24 00:00:00+00:00 | 22.05953979 | 23.09283066 | 21.72311974 | 22.99670982 | 22.29164696 | 8631377  | 3.904837418   | 0         | 0.726234355      |
| 2016-03-28 00:00:00+00:00 | 23.23700905 | 23.64551926 | 22.7564106  | 23.3090992  | 22.59445763 | 6867124  | 2.666666667   | 1         | 0.416691667      |
| 2016-03-29 00:00:00+00:00 | 22.94865036 | 23.42925072 | 22.51610947 | 23.35716057 | 22.64104843 | 10239950 | 2.682923525   | 0         | 0.420755881      |

If News_flag is 0, it means that there is no news on that day, and its sentiment value will be obtained by exponentially decaying the value of the last previous day with news, details can be found in the code
