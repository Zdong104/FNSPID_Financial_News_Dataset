# Data scraper
`data_scraper` aims to collect news and price data on stocks
`data_scraper` is divided into 3 main steps:
1. use headline scraper to download the headline and URL of the news
2. use the news content scraper to download the text of the news
3. use stock price scraper to download stock price

Before using it, please install `Selenium` and the corresponding [browser driver](https://googlechromelabs.github.io/chrome-for-testing/), and then follow the `requirements.txt` to configure the environment

`pip install -r requirements.txt`

## 1. headline scraper：
### Before running the scraper, make sure that your current working directory is the `headline_scraper` folder

`cd path/data_scraper/headline_scraper`

 1. Put all the lists in `list_original` into the `headline_scraper/lists` folder
 2. For the first time of use, please run `initialize_lists.py` to initialize list files
The `list_original` folder contains stock symbols starting with a-z in alphabetical order

 3. run `find_headlines v0.2.py`
 4. Type `a-z`, `a` for `list_a.csv` and `abc` for `list_a.csv`, `list_b.csv`, `list_c.csv`




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
6. Scraped text will be stored as `aapl.csv`, `ibm.csv` in `news_content_scraper/news_contents` folder

## 3. stock price scraper:
### Before running the scraper, make sure that your current working directory is the `stock_price_scraper` folder

`cd path/data_scraper/stock_price_scraper`
1. Put all the lists in `list_original` into the `stock_price_scraper/lists` folder
2. run 'add_Mark.py' first
4. run `get_price_from_yahoo.py`
5. Type `a-z`, as mentioned above
6. Scraped price data will be stored as `aapl.csv`, `ibm.csv` in `stock_price_scraper/yahoo` folder
