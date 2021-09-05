import os
import csv
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv

INPUT_CSV = os.path.join(os.getcwd(), "test_input.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "test_output.csv")

NN_model = Sequential()

# Input Model
NN_model.add(Dense(9, input_dim = 9, activation='linear'))

# Hidden Layers
NN_model.add(Dense(15, activation='linear'))
NN_model.add(Dense(12, activation='linear'))
NN_model.add(Dense(8, activation='linear'))
NN_model.add(Dense(6, activation='linear'))

# Middle Layer
NN_model.add(Dense(3, activation='linear'))

# Hidden Layers
NN_model.add(Dense(6, activation='linear'))
NN_model.add(Dense(8, activation='linear'))
NN_model.add(Dense(12, activation='linear'))
NN_model.add(Dense(15, activation='linear'))

# Output Layer
NN_model.add(Dense(9, activation='linear'))

NN_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
NN_model.summary()

weights_file = os.path.join(os.getcwd(), "trained_models", "model1", "Weights-071--0.17461.hdf5")
NN_model.load_weights(weights_file)
NN_model.compile(loss='mae', optimizer='adam', metrics=['mae'])

input_data = read_csv(INPUT_CSV, header=None)
predictions = NN_model.predict(input_data)
predictions = [['{0:.5f}'.format(x.item()) for x in y] for y in predictions]
with open(OUTPUT_CSV, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    for row in predictions:
        csv_writer.writerow(row)
