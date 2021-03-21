from tensorflow.keras.layers import Dense, Dropout, Conv2D, ReLU, LeakyReLU, Conv2DTranspose, BatchNormalization, Input, \
    Flatten, Lambda, Reshape, Activation
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# https://pravn.wordpress.com/category/vae-gan-vaegan/


class VAE:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_dim = (128, 128, 3)

        self.z_dim = 100
        self.build_model()

    def build_model(self):
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for f, k, s in zip([32, 64, 64, 64], [3, 3, 3, 3], [2, 2, 2, 2]):
            x = Conv2D(filters=f, kernel_size=k, strides=s, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for f, k, s in zip([64, 64, 32, 3], [3, 3, 3, 3], [2, 2, 2, 2]):
            x = Conv2DTranspose(filters=f, kernel_size=k, strides=s, padding='same')(x)

            if f != 3:
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                x = Dropout(rate=0.25)(x)
            else:
                x = Activation('tanh')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
        self.model.summary()


vae = VAE(8)
