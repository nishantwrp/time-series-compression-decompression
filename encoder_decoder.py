import os
import csv
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv

NN_model = Sequential()

# Input Model
NN_model.add(Dense(9, input_dim = 9, activation='linear'))

# Hidden Layers
NN_model.add(Dense(8, activation='linear'))
NN_model.add(Dense(6, activation='linear'))
NN_model.add(Dense(4, activation='linear'))

# Middle Layer
NN_model.add(Dense(3, activation='linear'))

# Hidden Layers
NN_model.add(Dense(4, activation='linear'))
NN_model.add(Dense(6, activation='linear'))
NN_model.add(Dense(7, activation='linear'))
NN_model.add(Dense(8, activation='linear'))
NN_model.add(Dense(9, activation='linear'))
NN_model.add(Dense(10, activation='linear'))
NN_model.add(Dense(12, activation='linear'))

# Output Layer
NN_model.add(Dense(9, activation='linear'))

NN_model.compile(loss='mae', optimizer='adam', metrics=['mae'])
NN_model.summary()

weights_file = "Weights-049--0.17903.hdf5"
NN_model.load_weights(weights_file)
NN_model.compile(loss='mae', optimizer='adam', metrics=['mae'])

encoder_weights = list()
encoder_bias = list()
encoder_layers = 5
decoder_weights = list()
decoder_bias = list()
decoder_layers = 9

counter = 0
# Weights and biases of the layers after training the model
for layer in NN_model.layers:
  if counter < encoder_layers:
    encoder_weights.append(layer.get_weights()[0])
    encoder_bias.append(layer.get_weights()[1])
  else:
    decoder_weights.append(layer.get_weights()[0])
    decoder_bias.append(layer.get_weights()[1])
  counter += 1

# Encoder layer
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
NN_encoder.summary()

counter = 0
for layer in NN_encoder.layers:
  layer.set_weights([encoder_weights[counter], encoder_bias[counter]])
  counter += 1

NN_encoder.save_weights("encoder_weights.hdf5")

# Decoder layer
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
NN_decoder.summary()

counter = 0
for layer in NN_decoder.layers:
  layer.set_weights([decoder_weights[counter], decoder_bias[counter]])
  counter += 1

NN_decoder.save_weights("decoder_weights.hdf5")
