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
