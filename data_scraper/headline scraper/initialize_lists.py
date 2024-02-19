import os

import pandas as pd

lists_folder = r"lists"
lists_saving_folder = r"lists"


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
    list_df_filtered["Desired_page"] = 0

    # 保存过滤后的数据到新的CSV文件
    list_df_filtered.to_csv(list_saving_path, index=False)

