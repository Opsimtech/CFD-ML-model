
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_cnn_lstm(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(128, activation='relu'), input_shape=input_shape))
    model.add(layers.LSTM(64, activation='relu'))
    model.add(layers.Dense(np.prod(output_shape)))
    model.add(layers.Reshape(output_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_convlstm(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same",
                                 return_sequences=False, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(filters=int(np.prod(output_shape)), kernel_size=(3, 3, 3),
                            activation='linear', padding='same'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_fno(input_shape, output_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(np.prod(output_shape)))
    model.add(layers.Reshape(output_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
