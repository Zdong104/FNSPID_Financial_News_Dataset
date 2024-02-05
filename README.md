# FNSPID
# Data scraper
Data scraper is divided into 3 main steps:
1. use headline scraper to scrape the headline and url of the news
2. use the news content scraper to crawl the text of the news
3. use stock price scraper to scrape stock price

Before using it, please install `Selenium` and the corresponding browser driver, and then follow the `requirements.txt` to configure the environment

`pip install -r requirements.txt`

## 1. Headline Scraper：
 Before running the scraper, make sure that your current working directory is the `Headline_scraper` folder

`cd path/data_scraper/Headline_scraper`

 1. Put all the lists in `list_original` into the `Headline_scraper/lists` folder

The `list_original` folder contains stock symbols starting with a-z in alphabetical order
 2. run `find_headlines v0.2.py`
 3. Type `a-z`, `a` for `list_a.csv` and `abc` for `list_a.csv`, `list_b.csv`, `list_c.csv`




![1707098657333](https://github.com/Zdong104/FNSPID/assets/91862936/9db14d61-9d44-4bcf-89d9-282de88238fd)

 4. Processed headlines will be stored in the `Headline_scraper/headlines` folder according to the stock symbol, e.g.: `aapl.csv`, `ibm.csv` ...

# 2. News content scraper:
 Before running the scraper, make sure that your current working directory is the `News_content_scraper` folder

`cd path/data_scraper/News_content_scraper`
1. Put all the lists in `list_original` into the `News_content_scraper/lists` folder
2. Put all files in `Headline_scraper/headlines` into `News_content_scraper/news_contents`
3. run 'add_Mark.py' first
4. run `find_content v0.2.py`
5. Type `a-z`, as mentioned above
6. Processed headlines will be stored in the `News_content_scraper/news_contents` folder
