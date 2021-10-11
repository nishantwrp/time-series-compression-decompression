import os
import csv
from keras.models import Sequential
from pandas import read_csv
from keras.layers import Dense

NN_decoder = Sequential()

# Middle Layer
NN_decoder.add(Dense(4, input_dim = 3, activation='linear'))
NN_decoder.add(Dense(6, activation='linear'))
NN_decoder.add(Dense(7, activation='linear'))
NN_decoder.add(Dense(8, activation='linear'))
NN_decoder.add(Dense(9, activation='linear'))
NN_decoder.add(Dense(10, activation='linear'))
NN_decoder.add(Dense(12, activation='linear'))

# Output Layer
NN_decoder.add(Dense(9, activation='linear'))

NN_decoder.compile(loss='mae', optimizer='adam', metrics=['mae'])
NN_decoder.load_weights("decoder_weights.hdf5")
NN_decoder.compile(loss='mae', optimizer='adam', metrics=['mae'])

INPUT_CSV = os.path.join(os.getcwd(), "test_output_mid_layer.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "final_layer_output.csv")

input_data = read_csv(INPUT_CSV, header=None)
predictions = NN_decoder.predict(input_data)
predictions = [['{0:.5f}'.format(x.item()) for x in y] for y in predictions]
with open(OUTPUT_CSV, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    for row in predictions:
        csv_writer.writerow(row)
