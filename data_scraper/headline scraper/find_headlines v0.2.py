import os
import random
import time

import pandas as pd

from selenium import webdriver
from selenium.common import TimeoutException, NoSuchElementException, StaleElementReferenceException, \
    ElementClickInterceptedException, NoSuchWindowException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/93.0.961.47 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
]

def get_webdriver():
    random_user_agent = random.choice(user_agents)
    headers = {'User-Agent': random_user_agent}
    options = Options()
    options.add_argument(f"user-agent={random_user_agent}")
    options.page_load_strategy = 'none'
    options.add_argument('--ignore-certificate-errors')
    # options.add_argument("--headless")
    options.add_argument('--log-level=3')
    # 禁用图片加载
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option(
        'excludeSwitches', ['enable-logging'])
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.cache.disk.enable", False)
    profile.set_preference("browser.cache.memory.enable", False)
    profile.set_preference("browser.cache.offline.enable", False)
    profile.set_preference("network.http.use-cache", False)
    options.profile = profile

    driver = webdriver.Chrome(options=options)
    return driver


def safe_find_element(driver, by, value):
    """安全地查找元素，处理可能的StaleElementReferenceException."""
    try:
        content = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((by, value)))
        return content
    except StaleElementReferenceException:
        time.sleep(2)
        content = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((by, value)))
        return content


def safe_find_elements(driver, by, value):
    """安全地查找元素，处理可能的StaleElementReferenceException."""
    try:
        content = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((by, value)))
        return content
    except StaleElementReferenceException:
        time.sleep(2)
        content = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((by, value)))
        return content


def find_headlines(driver, headline_file_path, headline_url, desired_page, list_df, index):
    # 'aapl'

    driver.delete_all_cookies()
    driver.execute_cdp_cmd("Network.setBlockedURLs", {
        "urls": ["*.flv*", "*.png", "*.jpg*", "*.jepg*", "*.gif*"]
    })
    # driver.minimize_window()
    now_retry = 0
    max_retries = 3
    headlines = []
    while True:
        if now_retry == max_retries:
            return desired_page
        try:
            driver.get(headline_url)
            # driver.minimize_window()

            attempts = 0

            while attempts < 2:
                try:
                    # 设置检测是否访问成功的标志，此处是加载其logo
                    flag_loaded = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'nsdq-logo--default'))
                    )
                    print("Loaded Logo")
                    break
                except NoSuchElementException:
                    attempts += 1
                    print(attempts, "times meet NoSuchElementException, Maybe meet Special Page")
            current_page = 0
            while True:
                headlines = []
                next_pages = WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'pagination__page')))
                exist_next = len(next_pages) > 1
                if desired_page == 9999:
                    print("This stock has been done")
                    break
                if current_page < desired_page:
                    next_page = WebDriverWait(driver, 20).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, 'pagination__next')))
                    driver.execute_script("arguments[0].click();", next_page)
                    current_page += 1
                    if next_page.get_attribute("disabled") == "true":
                        print("Last page reached.")
                        break
                    continue
                else:
                    # 如果老是断，就改大一点，2s比较稳定
                    time.sleep(2)

                # 加载headlines
                # headlines = WebDriverWait(driver, 20).until(
                #     EC.presence_of_all_elements_located((By.CLASS_NAME, 'quote-news-headlines__item')))
                headlines = safe_find_elements(driver, By.CLASS_NAME, 'quote-news-headlines__item')

                active_page_element = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'pagination__page--active')))
                current_page = int(active_page_element.text)

                for headline in headlines:
                    # 获取 headline的日期和内容
                    # headline_date = headline.find_element(By.XPATH, './/a/span')
                    headline_date = WebDriverWait(headline, 20).until(
                        EC.presence_of_element_located((By.XPATH, './/a/span')))

                    # headline_content = headline.find_element(By.XPATH, './/a/p')
                    headline_content = WebDriverWait(headline, 20).until(
                        EC.presence_of_element_located((By.XPATH, './/a/p')))

                    # headline_url = headline.find_element(By.XPATH, './/a').get_attribute('href')
                    headline_url = WebDriverWait(headline, 20).until(
                        EC.presence_of_element_located((By.XPATH, './/a'))).get_attribute('href')

                    headline_content_cleaned = headline_content.text.replace(u"\u2018", "'").replace(u"\u2019", "'")

                    data = {
                        'Date': [headline_date.text],
                        'Headline': [headline_content_cleaned],
                        'URL': [headline_url]
                    }

                    # 创建一个DataFrame
                    df = pd.DataFrame(data)
                    # 检查文件是否存在并且非空
                    file_exists = os.path.isfile(headline_file_path) and os.path.getsize(headline_file_path) > 0
                    # 然后根据文件是否已存在来决定是否写入列名
                    df.to_csv(headline_file_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                    list_df.at[index, 'Desired_page'] = desired_page
                    list_df.to_csv(list_file_path, index=False)
                # 翻页按钮
                next_page = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'pagination__next')))
                # next_page = next_pages[0]
                # 只有一页
                if not exist_next:
                    print("Only one page")
                    desired_page = 9999
                    break
                if next_page.get_attribute("disabled") == "true":
                    print("Last page reached.")
                    desired_page = 9999
                    break
                print("current_page:", current_page)
                desired_page = current_page

                # Click the next page button
                driver.execute_script("arguments[0].click();", next_page)
            return desired_page
        except TimeoutException:
            try:
                # 查找页面中的<h1>标签 判断是否是特殊情况
                h2_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'alert__heading')))
                # 找到匹配项后退出循环
                for h2 in h2_elements:
                    if h2.text.find("trading") != -1:
                        print("time out, finding alerts")
                        print("Alert found")
                        desired_page = 9999
                        return desired_page
            except TimeoutException:
                print("Time out or wrong url")
                return desired_page
        except ElementClickInterceptedException:
            print("button missed, desired_page:", desired_page)
            return desired_page
        except NoSuchWindowException:
            print("closed window manually, desired_page:", desired_page)
            print("5 sec later, driver will restart, if you want to TERMINATE this process please be quick")
            time.sleep(15)
            driver = get_webdriver()
            return desired_page
        except StaleElementReferenceException:
            now_retry += 1
            print("maybe too fast, elements weren't ready, desired_page:", desired_page)
            try:
                WebDriverWait(driver, 20).until(
                    EC.staleness_of(headlines[0])
                )
            except Exception:
                pass
            continue
            # return desired_page


def start_find(list_df):
    for index, row in list_df.iloc[start_row_index:].iterrows():
        print("start: ", row["Stock_name"], "from:", row["Desired_page"])
        if row["Desired_page"] == 9999:
            print("This stock Has been done")
            continue
        driver = get_webdriver()
        # print("index: ", index)
        headline_file_path = "headlines/" + str(row['Stock_name']) + ".csv"
        headlines_url = "https://www.nasdaq.com/market-activity/stocks/" + str(row['Stock_name']) + "/news-headlines"
        Desired_page = find_headlines(driver, headline_file_path, headlines_url, row['Desired_page'], list_df, index)
        driver.close()
        list_df.at[index, 'Desired_page'] = Desired_page
        print("now desired_page: ", list_df.loc[row.name, 'Desired_page'])
        list_df.to_csv(list_file_path, index=False)
        print(list_df)
    return list_df


if __name__ == "__main__":
    # find_headlines("aapl")
    inits = input("from which list:")
    start_row_index = 0
    for init in inits:
        list_file_path = 'lists/list_' + init + '.csv'
        if not os.path.isfile(list_file_path):
            print("!!! Wrong list_name !!!")
        else:
            list_df = pd.read_csv(list_file_path, encoding="utf-8")
            while not list_df['Desired_page'].isin([9999]).all():
                print("need find")
                list_df = start_find(list_df)
            print("list_", init, "completed")

    # 传入参数就是股票的代称
