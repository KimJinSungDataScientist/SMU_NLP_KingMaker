import pandas as pd
import numpy as np
import cv2
import sys, os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, layers, losses

class Autoencoder_cnn(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder_cnn, self).__init__()
    self.latent_dim = encoding_dim
    self.encoder = tf.keras.Sequential([
      layers.Conv2D(8,kernel_size=(3,3),strides=(2,2),padding='same',activation='relu'),
      layers.Conv2D(3,kernel_size=(3,3),strides=(2,2),padding='same',activation='sigmoid'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=(3,3),strides=(2,2),padding='same',activation='relu'),
      layers.Conv2DTranspose(3, kernel_size=(3,3),strides=(2,2),padding='same',activation='sigmoid'),
      # layers.Dense(80*80*3, activation='sigmoid'),
      # layers.Reshape((80, 80,3))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def vec_extract(self, x):
    encoded = self.encoder(x)
    return encoded

  def vec_img(self, x):
    decoded = self.decoder(x)
    return decoded

class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = encoding_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(1024, activation='sigmoid'),
      layers.Dense(256, activation='sigmoid'),
      layers.Dense(self.latent_dim, activation='sigmoid'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(256, activation='sigmoid'),
      layers.Dense(1024, activation='sigmoid'),
      layers.Dense(80*80*3, activation='sigmoid'),
      layers.Reshape((80, 80,3))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def vec_extract(self, x):
    encoded = self.encoder(x)
    return encoded

  def vec_img(self, x):
    decoded = self.decoder(x)
    return decoded


class U_net(Model):
    def __init__(self, img_shape):
        super(U_net, self).__init__()
        self.input_shape_ = img_shape
        self._build()

    def _build(self):
        input_ = layers.Input(shape=self.input_shape_, name='input')
        x = input_  # 80,80,3
        # c0 = layers.Conv2D(32,kernel_size=(1,1),activation='relu')(x)
        # x = c0
        x = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 78,78,64
        c1 = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 76,76,64
        x = c1
        x = layers.MaxPool2D()(x)  # 38,38,64

        x = layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 36,36,128
        c2 = layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 34,34,128
        x = c2
        x = layers.MaxPool2D()(x)  # 17,17,128

        x = layers.Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 15,15,256
        x = layers.Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 13,13,256

        x = layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
            x)  # 26,26,128
        c2 = layers.Cropping2D(cropping=((4, 4), (4, 4)))(c2)  # 26,26,128
        x = layers.Concatenate(axis=3)([x, c2])  # 26,26,256

        x = layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 24,24,128
        x = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 22,22,64

        x = layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
            x)  # 44,44,64
        c1 = layers.Cropping2D(cropping=((16, 16), (16, 16)))(c1)  # 44,44,64
        x = layers.Concatenate(axis=3)([x, c1])  # 44,44,128

        x = layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 42,42,32
        x = layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu')(x)  # 40,40,32

        x = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(
            x)  # 80,80,32
        # x = layers.Concatenate(axis=3)([x,c0]) # 80,80,64
        out = layers.Conv2D(3, kernel_size=(1, 1), activation='sigmoid')(x)

        model_input = input_
        model = Model(model_input, out)
        self.model = model

    def compile(self):
        self.model.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def call(self, x):
        out = self.model(x)
        return out

    def load_trained_model(self, model):
        self.model = model
        
    def train(self, dataset, epochs=15):
        self.model.fit(dataset, epochs=epochs)

    def vec_extract(self, x):
        fe_layer = self.model.layers[:9]
        for layer in fe_layer:
            x = layer(x)
        out = x
        out = layers.Conv2D(1, 1, kernel_initializer=tf.keras.initializers.ones)(x)
        out = out.numpy()
        out = (out.max() - out) / (out.max() - out.min())
        # out = layers.GlobalAveragePooling2D()(x)
        return out