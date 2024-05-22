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