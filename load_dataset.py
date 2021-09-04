import os
from pandas import read_csv
from sklearn.model_selection import train_test_split

FINAL_DATASET_DIRECTORY = os.path.join(os.getcwd(), "processed_dataset")
DATASET_NAME = "dataset.csv"
PROCESSED_DATASET_PATH = os.path.join(FINAL_DATASET_DIRECTORY, DATASET_NAME)

TEST_FRACTION = 0.05

def load_dataset():
    df = read_csv(PROCESSED_DATASET_PATH, header=None)
    train, test = train_test_split(df, test_size=TEST_FRACTION)
    return train, test
