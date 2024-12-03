# You can download the dataset by running this script
# Remember to pip install datasets before running this script
from datasets import load_dataset

# Download the dataset and save to the current directory
dataset = load_dataset("Zihan1004/FNSPID", cache_dir=".")
