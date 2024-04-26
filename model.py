import tensorflow as tf
import numpy as np
import keras
from keras import Sequential, layers
from keras.activations import softmax, tanh
from keras.initializers import Identity
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPooling1D,
)
from tensorflow.python.keras import backend as K


class Dain(layers.Layer):
    def __init__(self, dim, n_features, name="dain", **kwargs):
        super(Dain, self).__init__(name=name, **kwargs)
        self.dim = dim
        self.n_features = n_features
        self.eps = 1e-8

        self.mean_layer = layers.Dense(
            n_features, use_bias=False, kernel_initializer=Identity, name="dain-mean"
        )
        self.scaling_layer = layers.Dense(
            n_features, use_bias=False, kernel_initializer=Identity, name="dain-scale"
        )
        self.gating_layer = layers.Dense(
            n_features, activation="sigmoid", name="dain-gate"
        )

        self.transpose = layers.Permute((2, 1))
        self.reshape_2d = layers.Reshape((dim, n_features))

    @staticmethod
    def fn(elem):
        return K.switch(K.less_equal(elem, 1.0), K.ones_like(elem), elem)

    def call(self, inputs):
        # step 1: adapative average
        # from (batch, rows, n_features) to (batch, n_features, rows)
        inputs = self.transpose(inputs)
        avg = K.mean(inputs, axis=2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = K.reshape(adaptive_avg, (-1, self.n_features, 1))
        inputs -= adaptive_avg

        # # step 2: adapative scaling
        std = K.mean(inputs ** 2, axis=2)
        std = K.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std = K.map_fn(self.fn, adaptive_std)
        adaptive_std = K.reshape(adaptive_std, (-1, self.n_features, 1))
        inputs /= adaptive_std

        # # step 3: gating
        avg = K.mean(inputs, axis=2)
        gate = self.gating_layer(avg)
        gate = K.reshape(gate, (-1, self.n_features, 1))
        inputs *= gate
        # from (batch, n_features, rows) => (batch, rows, n_features)
        inputs = self.transpose(inputs)
        return inputs

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Dain, self).build(input_shape)


class Attention(Layer):

    def __init__(self, units, name="attention", **kwargs):
        super(Attention, self).__init__(name=name, **kwargs)
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        # Compute attention scores
        score = tanh(self.W(inputs))
        attention_weights = softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def build(self, input_shape):
        self.W.build(input_shape)
        self.V.build(input_shape)
        self.built = True

def generate_model():
    N = 150
    cnn_lstm_model = Sequential()

    cnn_lstm_model.add(Input(shape=(N, 2)))
    cnn_lstm_model.add(Dain(N, 2))
    cnn_lstm_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    cnn_lstm_model.add(MaxPooling1D(pool_size=3))
    cnn_lstm_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_lstm_model.add(MaxPooling1D(pool_size=2))

    cnn_lstm_model.add(LSTM(units=256, return_sequences=True))
    cnn_lstm_model.add(Dropout(0.3))
    cnn_lstm_model.add(LSTM(units=128, return_sequences=True))
    cnn_lstm_model.add(Dropout(0.3))
    cnn_lstm_model.add(Attention(128))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Flatten())
    cnn_lstm_model.add(Dense(128, activation='relu'))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Dense(64, activation='relu'))
    cnn_lstm_model.add(Dropout(0.2))
    cnn_lstm_model.add(Dense(5, activation='softmax'))
    return cnn_lstm_model


def __adjust_array(arr, target_length=150):
    # Get the current number of rows in the array
    current_length = arr.shape[0]
    
    if current_length > target_length:
        # Truncate the array if it's longer than the target
        adjusted_array = arr[:target_length]
    elif current_length < target_length:
        # Calculate the number of rows to pad
        rows_to_add = target_length - current_length
        # Get the number of columns in the array
        num_columns = arr.shape[1]
        # Create an array of zeros to pad
        padding = np.zeros((rows_to_add, num_columns), dtype=arr.dtype)
        # Append the padding to the original array
        adjusted_array = np.vstack((arr, padding))
    else:
        # If the length is already target_length, return the original array
        adjusted_array = arr
    
    return adjusted_array

def prepare_input(fovs):
    input_trajs = []
    for fov in fovs:
        for traj_ind in range(len(fov)):
                input_trajs.append(__adjust_array(fov[traj_ind]))
    return np.array(input_trajs)

def get_preds(model_output):
    return np.argmax(model_output, axis=1)
