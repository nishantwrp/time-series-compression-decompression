import os
import csv

NUM_PERSONS = 20
DIGITS = [i for i in range(0, 5)]
NUM_FILES = 50
NUM_SENSORS = 9
MAX_DATA_POINTS = 100

NUM_COLUMNS_IN_FINAL_DATASET = NUM_SENSORS*MAX_DATA_POINTS
FINAL_DATASET_DIRECTORY = os.path.join(os.getcwd(), "processed_dataset")

if not os.path.exists(FINAL_DATASET_DIRECTORY):
    os.makedirs(FINAL_DATASET_DIRECTORY)

DATASET_NAME = "dataset_5_digit_20_people_50_files.csv"
PROCESSED_DATASET_PATH = os.path.join(FINAL_DATASET_DIRECTORY, DATASET_NAME)

all_rows = []

def process_dataset(path):
    with open(path, "r") as f:
        file_content = f.readlines()

        for line in range(1, MAX_DATA_POINTS+1):
            all_rows.append(file_content[line].strip().split(","))

for person in range(1, NUM_PERSONS+1):
    for digit in DIGITS:
        for file in range(1, NUM_FILES+1):
            file_path = os.path.join(os.getcwd(), "raw_dataset", "P%s" % person, "D%s" % digit, "%s_D%s.csv" % (file, digit))
            process_dataset(file_path)

with open(PROCESSED_DATASET_PATH, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    for row in all_rows:
        csv_writer.writerow(row)
