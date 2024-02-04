import os

import pandas as pd

lists_folder = r"lists"
lists_saving_folder = r"lists"
news_contents_folder = r"news_contents"
news_contents_saving_folder = r"news_contents"


# process lists
print("---Lists---")
list_files = [file for file in os.listdir(lists_folder) if file.endswith('.csv')]
for list_file in list_files:
    print(list_file)
    list_file_path = os.path.join(lists_folder, list_file)
    list_saving_path = os.path.join(lists_saving_folder, list_file)

    list_df = pd.read_csv(list_file_path, encoding="utf-8", on_bad_lines="skip")

    list_df.columns = list_df.columns.str.capitalize()
    list_df_filtered = list_df[["Stock_name"]]
    list_df_filtered["Mark"] = 0

    # 保存过滤后的数据到新的CSV文件
    list_df_filtered.to_csv(list_saving_path, index=False)

# process new_content files
print("---News---")
news_content_files = [file for file in os.listdir(news_contents_folder) if file.endswith('.csv')]
for news_content_file in news_content_files:
    print(news_content_file)
    news_content_file_path = os.path.join(news_contents_folder, news_content_file)
    news_content_saving_path = os.path.join(news_contents_saving_folder, news_content_file)
    try:
        news_content_df = pd.read_csv(news_content_file_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        news_content_df = pd.read_csv(news_content_file_path, encoding="ISO-8859-1", on_bad_lines="skip")

    news_content_df.columns = news_content_df.columns.str.capitalize()
    news_content_df["Mark"] = 0
    news_content_df["Text"] = "0"
    news_content_df.drop_duplicates(subset="Url")
    news_content_df_filtered = news_content_df[["Date", "Url", "Text", "Mark"]]

    # 保存过滤后的数据到新的CSV文件
    news_content_df_filtered.to_csv(news_content_saving_path, index=False)
