import os
import csv
from keras.models import Sequential
from pandas import read_csv
from keras.layers import Dense

NN_encoder = Sequential()

# Input Model
NN_encoder.add(Dense(9, input_dim = 9, activation='linear'))

# Hidden Layers
NN_encoder.add(Dense(8, activation='linear'))
NN_encoder.add(Dense(6, activation='linear'))
NN_encoder.add(Dense(4, activation='linear'))

# Output Layer
NN_encoder.add(Dense(3, activation='linear'))

NN_encoder.compile(loss='mae', optimizer='adam', metrics=['mae'])
NN_encoder.load_weights("encoder_weights.hdf5")
NN_encoder.compile(loss='mae', optimizer='adam', metrics=['mae'])

INPUT_CSV = os.path.join(os.getcwd(), "test_input.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "test_output_mid_layer.csv")

input_data = read_csv(INPUT_CSV, header=None)
predictions = NN_encoder.predict(input_data)
predictions = [['{0:.5f}'.format(x.item()) for x in y] for y in predictions]
with open(OUTPUT_CSV, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    for row in predictions:
        csv_writer.writerow(row)
