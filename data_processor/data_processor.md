![image](https://github.com/Zdong104/FNSPID/assets/91862936/566fb249-8392-401f-be5b-140bcbcc8284)# Data processor
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
