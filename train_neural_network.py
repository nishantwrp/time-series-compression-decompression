from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from load_dataset import load_dataset

class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()

    def add_dense_input_layer(self, units, input_dim, activation):
        self.model.add(Dense(units, input_dim=input_dim, activation=activation))

    def add_dense_hidden_layer(self, units, activation):
        self.model.add(Dense(units=units, activation=activation))

    def set_compilation_args(self, loss, optimizer, metrics):
        self.model_loss = loss
        self.model_optimizer = optimizer
        self.model_metrics = metrics

    def compile_model(self):
        self.model.compile(loss=self.model_loss, optimizer=self.model_optimizer, metrics=self.model_metrics)

    def print_model_summary(self):
        print(self.model.summary())

    def create_model_checkpoint_to_save_best(self):
        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
        return checkpoint

    def load_checkpoint_weights(self, filename):
        self.model.load_weights(filename)
        self.compile_model()

# Some notes about the current implementation
# 1. We currently decided on the "linear" activation method instead of something like widely use "ReLU" because these activation functions usually neglect the negative weights.
# 2. We can probably work on a better distribution of hidden layers for a optimal prediction.
# 3. We tested this code on Google Colab because of limited power of our local machines.
# 4. An acceptable metric needs to be discussed yet for the allowed error range for the calculation of accuracy of the generated time series finally.
train_dataset, test_dataset = load_dataset()
neural_network = NeuralNetwork()

neural_network.add_dense_input_layer(9, 9, "linear") # Input layer.
neural_network.add_dense_hidden_layer(15, "linear")
neural_network.add_dense_hidden_layer(12, "linear")
neural_network.add_dense_hidden_layer(8, "linear")
neural_network.add_dense_hidden_layer(6, "linear")
neural_network.add_dense_hidden_layer(3, "linear") # Middle layer. We'll break our model from here in phase 2.
neural_network.add_dense_hidden_layer(6, "linear")
neural_network.add_dense_hidden_layer(8, "linear")
neural_network.add_dense_hidden_layer(12, "linear")
neural_network.add_dense_hidden_layer(15, "linear")
neural_network.add_dense_hidden_layer(9, "linear") # Output Layer.

neural_network.set_compilation_args("mae", "adam", ["mae"])
neural_network.compile_model()

neural_network.print_model_summary()

save_best_callback = neural_network.create_model_checkpoint_to_save_best()
# This will save the the checkpoint with minimum loss alongside in a file with filename something like Weights-033--0.71620.hdf5. Ideally this checkpoint
# should be used to make predictions now.
neural_network.model.fit(train_dataset, train_dataset, epochs=100, batch_size=64, validation_split = 0.2, callbacks=[save_best_callback])

# Code to make a prediction. As stated above a metric to decide the allowed error range needs to be discussed.
weights_file = 'Weights-033--0.71620.hdf5' # EDIT THIS
neural_network.load_weights(weights_file)
neural_network.compile_model()

prediction = neural_network.model.predict([list(test_dataset.iloc[0])])
print(prediction)
