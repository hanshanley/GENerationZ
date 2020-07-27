# -*- coding: utf-8 -*-
"""
Created on  June 18th

@author: hanshanley

Convolutional variational autoencoder for training a model that projects 
chemical compounds to a latent space. This particular model further has a
4-layer neural network that takes points in the latent space and to predict 
a compounds properties. Specifically it takes the latent space and information
about a target's rna sequences and uses this information to predict the IC50 
value. IC50 is half maximal inhibitory concentration (IC50) is a measure of 
the effectiveness of a substance in inhibiting a specific biological or
biochemical function.

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import pandas as pd
import math
import tensorflow.keras.layers as layers
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
HIDDEN_DIM = 256

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size,embedding_dim, 
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
    
  def call(self, x):
    x = self.embed(x)
    x = self.conv1(x)
    x =  layers.BatchNormalization(axis = -1)(x)
    for i in range(len(self.conv_layers)):
      x = self.conv_layers[i](x)
    x =  layers.BatchNormalization(axis = -1)(x)
    x = layers.Flatten()(x)

    x = self.dense1(x)
    x = self.drop1(x)
    x =  layers.BatchNormalization(axis = -1)(x)
    z_mean = self.mean(x)
    z_log_var = self.log_var(x)
    return x, z_mean, z_log_var

  ## Used to sample from the latent space
  def sample(self,z):
    z_mean,z_log_var = z
    batch_size = z_mean.shape[0]
    epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
    return z_mean + K.exp(z_log_var/2)*epsilon

class Decoder(tf.keras.layers.Layer):
  def __init__(self,embedding_dim, vocab_size, dropout_rate, max_len,latent_dim):
    super(Decoder, self).__init__()
    self.dense1 = layers.Dense(latent_dim*4)
    self.drop1 = layers.Dropout(dropout_rate)
    self.rv = tf.keras.layers.RepeatVector(max_len)
    self.lstm1 = tf.keras.layers.LSTM(embedding_dim,return_sequences = True,activation ='tanh')
    self.timeD = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))

  def call(self, x):
    x = self.dense1(x)
    x = self.drop1(x)
    x = self.rv(x)
    x = self.lstm1(x)
    x = self.timeD(x)
    return x

## Auxiliary Network 
class NU_z(tf.keras.layers.Layer):
  def __init__(self, inter_size):
    super(NU_z, self).__init__()
    self.dense1 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense2 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense3 = tf.keras.layers.Dense(inter_size,activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    self.dense4  = tf.keras.layers.Dense(1)

  def call(self, z_x,genes):
    pred = tf.concat([z_x,genes],axis = -1)
    pred = self.dense1(pred)
    pred = self.dense2(pred)
    pred = self.dense3(pred)
    pred = self.dense4(pred)
    return pred

def softmax_logits_loss_with_pad(labels,logits):
  weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
  nonpad_seq = tf.math.count_nonzero(weights, dtype=tf.dtypes.float32, )
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits =logits)
  loss = loss *weights 
  return tf.reduce_sum(loss)

class SMILE_VAE(tf.keras.Model):
  def __init__(self, vocab_size,embedding_dim,
              max_len, latent_dim,
              recurrent_dropout =0.2,
              dropout_rate=0.2,
              epsilon_std = 1.0):
    super(SMILE_VAE, self).__init__()
    self.encoder = Encoder(vocab_size,embedding_dim, 
                          max_len,latent_dim,
                          recurrent_dropout =0.2,
                          dropout_rate=0.2,
               epsilon_std = 1.0)
    self.decoder = Decoder(vocab_size = vocab_size, 
                           embedding_dim = 256, 
                           dropout_rate = dropout_rate, 
                           max_len = max_len,
                           latent_dim = latent_dim)
    self.latent_dim = latent_dim
    self.nu_z = NU_z(512)

  def call(self, x):
    h, z_mean,z_log_var = self.encoder(x)
    z = tf.keras.layers.Lambda(self.encoder.sample, output_shape =(self.latent_dim,))([z_mean,z_log_var])
    x_decoded = self.decoder(z)

    ## Returns latent space encoding, its variance, and the decoded
    ## version of the space encoding. 
    return z_mean,z_log_var, x_decoded

  ## Return variational loss: cross entropy + KL loss 
  def vae_loss(self,labels,x_decoded,z_mean,z_log_var,beta):
    x_ent_loss = softmax_logits_loss_with_pad(labels = labels, logits = x_decoded)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return tf.reduce_sum(x_ent_loss +  beta*tf.reduce_mean(kl_loss ))