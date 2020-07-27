# -*- coding: utf-8 -*-
"""
Created on  June 18th

@author: hanshanley

Implicit variational autoencoder for training a model that projects 
chemical compounds to a latent space. This model does not use a Guassian 
distirbution with a mean and variance determined by the encoder 
to generate the KL divergence but rather samples using a separate auxilary 
trained network. This methodology was seen by https://www.aclweb.org/anthology/D19-1407.pdf
as being effective in text generation. 

"""

import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow.keras.layers  as layers
import pandas as pd
import math
import time
import numpy as np
import matplotlib.pyplot as plt

BATCH_NORM = True
CONV_ACTIVATION = 'tanh'
CONV_DEPTH = 4
CONV_DIM_DEPTH = 32
CONV_DIM_WIDTH = 16
CONV_D_GF = 1.15875438383
CONV_W_GF = 1.1758149644
HIDDEN_DIM = 100
HG_GROWTH_FACTOR = 1.4928245388
MIDDLE_LAYERS = 1

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size,embedding_dim, hidden_dim,
               max_len,latent_dim,
               recurrent_dropout =0.2,
               dropout_rate=0.2,
               epsilon_std = 1.0):
    super(Encoder, self).__init__()

    self.lstm1 = tf.keras.layers.LSTM(256,
                                      return_sequences = False)
    self.latent_dim = latent_dim
    self.epsilon_std = epsilon_std
    self.drop1 = tf.keras.layers.Dropout(dropout_rate)
    self.drop2 = tf.keras.layers.Dropout(dropout_rate)
    self.mean = tf.keras.layers.Dense(latent_dim)
    self.log_var = tf.keras.layers.Dense(latent_dim)
    self.embed =  keras.layers.Embedding(input_dim=vocab_size, 
                                         output_dim=embedding_dim,
                                         mask_zero = True, 
                                embeddings_initializer='random_normal',
                                input_length=max_len,
                                trainable=True)
    self.conv1 = layers.Conv1D(int(CONV_DIM_DEPTH *CONV_D_GF),int(CONV_DIM_WIDTH*CONV_W_GF),
                         activation ='tanh')

    ## Convolutional layer that builds increasingly upwards to larger depths using a
    ## a given growth rate 
    self.conv_layers =  [layers.Conv1D(int(CONV_DIM_DEPTH *CONV_D_GF**j),int(CONV_DIM_WIDTH*CONV_W_GF**j),
                         activation ='tanh') for j in  range(1,CONV_DEPTH-1) ]
    self.dense1 = layers.Dense(latent_dim*4)
    
  def call(self, x, eps):
    x = self.embed(x)
    x = self.conv1(x)
    x =  layers.BatchNormalization(axis = -1)(x)
    for i in range(len(self.conv_layers)):
      x = self.conv_layers[i](x)
    x =  layers.BatchNormalization(axis = -1)(x)
    x = layers.Flatten()(x)

    x = self.dense1(x)
    x = self.drop1(x)
    enc =  layers.BatchNormalization(axis = -1)(x)
    z_mean = self.mean(tf.concat([enc],axis = 1))
    return  enc, z_mean

  ## Used to sample from the latent space 
  def sample(self,z):
    z_mean,z_log_var = z
    batch_size = z_mean.shape[0]
    epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
    return z_mean + K.exp(z_log_var/2)*epsilon


## Auxiliary network that takes in the latent space encoding 
## as well as the last layer output of an lstm
class NU_xz(tf.keras.layers.Layer):
  def __init__(self, inter_size):
    super(NU_xz, self).__init__()
    self.dense1 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense2 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense3 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense4  = tf.keras.layers.Dense(1)

  def call(self, z_x, enc):
    zs = tf.concat([z_x,enc], axis =1)
    zs = self.dense1(zs)
    zs = self.dense2(zs)
    zs = self.dense3(zs)
    zs = self.dense4(zs)

    return zs


## Auxiliary network that takes in the latent space encoding 
class NU_z(tf.keras.layers.Layer):
  def __init__(self, inter_size):
    super(NU_z, self).__init__()
    self.dense1 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense2 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense3 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense4  = tf.keras.layers.Dense(1)

  def call(self, z_x ):
    z = self.dense1(z_x)
    z = self.dense2(z)
    z = self.dense3(z)
    z = self.dense4(z)
    return z

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, hidden_dim,
               embedding_dim,
               latent_dim, max_len,
               recurrent_dropout =0.2,
               dropout_rate = 0.2,
               epsilon_std = 1.0):
    super(Decoder, self).__init__()

    self.hidden_dim = hidden_dim
    self.max_len = max_len
    self.latent_dim = latent_dim
    self.hidden_dense = tf.keras.layers.Dense(hidden_dim)
    self.rv = tf.keras.layers.RepeatVector(max_len-1)
    self.lstm1 = tf.keras.layers.LSTM(hidden_dim,return_sequences = True)
    self.timeD = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
    self.embed =  keras.layers.Embedding(input_dim=vocab_size, 
                                         output_dim=embedding_dim,
                                         mask_zero = True, 
                                embeddings_initializer='random_normal',
                                input_length=max_len,
                                trainable=True)
    self.dense1 = layers.Dense(latent_dim*4)
    self.drop1 = layers.Dropout(dropout_rate)
    self.drop2 = layers.Dropout(dropout_rate)

  def call(self,labels, z_x):
    x = self.rv(z_x)
    x = self.dense1(x)
    x = self.drop1(x)
    dec_input = x
    x = self.lstm1(dec_input)
    x = self.timeD(x)
    return x

class SMILE_IMPLICIT_VAE(tf.keras.Model):
  def __init__(self, vocab_size,embedding_dim, 
              max_len, latent_dim, hidden_dim,
              recurrent_dropout =0.2,
              dropout_rate=0.2,
              epsilon_std = 1.0):
    super(SMILE_IMPLICIT_VAE, self).__init__()
    self.encoder = Encoder(vocab_size = vocab_size,
                           embedding_dim = embedding_dim, hidden_dim = hidden_dim,
                          max_len = max_len,latent_dim = latent_dim,
                          recurrent_dropout =0.2,
                          dropout_rate=0.2,
                          epsilon_std = 1.0)
    self.decoder = Decoder(vocab_size = vocab_size, hidden_dim = hidden_dim,embedding_dim=384,
                          max_len = max_len,latent_dim = latent_dim,
                          recurrent_dropout =0.2,
                          dropout_rate=0.2,
                          epsilon_std = 1.0)
    self.latent_dim = latent_dim
    self.nu_xz = NU_xz(hidden_dim)
    self.nu_z = NU_z(hidden_dim)

  def call(self, x):
    eps = tf.convert_to_tensor(np.random.normal(size=(x.shape[0], self.latent_dim)),dtype = tf.float32)
    enc , z_x  = self.encoder(x,eps)
    x_decoded = self.decoder(z_x)

    ## Returns latent space encoding, its last layer encoding, and the decoded
    ## version of the space encoding
    return x_decoded, enc, z_x

  ## Auxiliary network loss: with latent space and last layer encoding
  def kl_xz_loss(self, z_x, enc):
    z = tf.convert_to_tensor(np.random.normal(size=(z_x.shape[0], self.latent_dim)),dtype = tf.float32)
    kl_xz = tf.reduce_mean(tf.keras.backend.exp(self.nu_xz(z_x= z, enc =enc)) - self.nu_xz(z_x = z_x, enc =enc)) 
    return kl_xz

  ## Auxiliary network loss: with latent space
  def kl_z_loss(self, z_x):
    z = tf.convert_to_tensor(np.random.normal(size=(z_x.shape[0], self.latent_dim)),dtype = tf.float32)
    kl_z = tf.reduce_mean(tf.keras.backend.exp(self.nu_z(z)) - self.nu_z(z_x))
    return kl_z