# -*- coding: utf-8 -*-
"""
Created on  June 18th

@author: hanshanley

Decoder Transformer that takes points in the latent space and 
atttempts to decode them back into smiles. This is just the decoder,
its takes encodings from pretrained variational encoders. This enables a more 
powerful decoder to be used to decode the smiles while allowing the latent 
space to still model the smiles and their properties. This decoder is based on 
https://ieeexplore.ieee.org/abstract/document/8852155

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, Dropout, Add, Input, Lambda
from tensorflow.keras.initializers import Ones, Zeros

def get_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def attention_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp):
  atn_mask = attention_mask(tf.shape(inp)[1])
  padding_mask = get_padding_mask(inp)
  mask = tf.maximum(padding_mask, atn_mask)

  return mask

def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc

def shape_list(x):
  tmp = K.int_shape(x)
  tmp = list(tmp)
  tmp[0] = -1
  return tmp

def split_heads(x, n: int, k: bool = False):  # B, L, C
  x_shape = shape_list(x)
  m = x_shape[-1]
  new_x_shape = x_shape[:-1] + [n, m // n]
  new_x = K.reshape(x, new_x_shape)
  return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])

def merge_heads(x):
  new_x = K.permute_dimensions(x, [0, 2, 1, 3])
  x_shape = shape_list(new_x)
  new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
  return K.reshape(new_x, new_x_shape)

def scaled_dot_product_attention(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):
  #w = K.batch_dot(q, k)  # w is B, H, L, L
  w = tf.matmul(q, k, transpose_b=False)
  w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
  if attn_mask is not None:
      w = attn_mask * w + (1.0 - attn_mask) * neg_inf
  w = K.softmax(w)
  w = Dropout(attention_dropout)(w)
  return  tf.matmul(w, v,transpose_b=False)  # it is B, H, L, C//H [like v]

def multihead_attention_imp( q, k, v, mask=None):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  if self.scale:
      dk = tf.cast(tf.shape(k)[-1], tf.float32)
      matmul_qk = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
      matmul_qk += (mask * -1e9)

  attention_weights = tf.nn.softmax(matmul_qk, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

def multihead_attention(x, attn_mask, n_head: int, n_state: int, attention_dropout: float, neg_inf: float):
  _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
  q = split_heads(_q, n_head)  # B, H, L, C//H
  k = split_heads(_k, n_head, k=True)  # B, H, C//H, L
  v = split_heads(_v, n_head)  # B, H, L, C//H
  a = scaled_dot_product_attention(q, k, v, attn_mask, attention_dropout, neg_inf)
  return merge_heads(a)

def gelu(x):
  return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, n_head: int, n_state: int, attention_dropout: float, use_attn_mask: bool, neg_inf: float,
                 **kwargs):  
    super().__init__(**kwargs)
    self.n_head = n_head
    self.n_state = n_state
    self.attention_dropout = attention_dropout
    self.use_attn_mask = use_attn_mask
    self.neg_inf = neg_inf

  def compute_output_shape(self, input_shape):
    x = input_shape[0] if self.use_attn_mask else input_shape
    return x[0], x[1], x[2] // 3

  def call(self, inputs):
    x = inputs[0] if self.use_attn_mask else inputs
    attn_mask = inputs[1] if self.use_attn_mask else None
    return multihead_attention(x, attn_mask, self.n_head, self.n_state, self.attention_dropout, self.neg_inf)

class LayerNormalization(tf.keras.layers.Layer):
  def __init__(self, eps: float = 1e-5):
    super(LayerNormalization, self).__init__()
    self.eps = eps

  def build(self, input_shape):
    self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
    self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
    super().build(input_shape)

  def call(self, x):
    u = K.mean(x, axis=-1, keepdims=True)
    s = K.mean(K.square(x - u), axis=-1, keepdims=True)
    z = (x - u) / K.sqrt(s + self.eps)
    return self.gamma * z + self.beta

  def compute_output_shape(self, input_shape):
    return input_shape

class Gelu(tf.keras.layers.Layer):
  def __init__(self, accurate: bool = False):
    super(Gelu, self).__init__()
    self.accurate = accurate

  def call(self, inputs):
    if not self.accurate:
        return gelu(inputs)
    erf = K.tf.erf
    return inputs * 0.5 * (1.0 + erf(inputs / math.sqrt(2.0)))

  def compute_output_shape(self, input_shape):
      return input_shape

class MultiHeadSelfAttention(tf.keras.layers.Layer):
  def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                use_attn_mask: bool, layer_id: int, neg_inf: float):
    super(MultiHeadSelfAttention, self).__init__()
    assert n_state % n_head == 0
    self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
    self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,
                                    neg_inf, name='layer_{}/self_attention'.format(layer_id))
    self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

  def call(self, x, training=True):
    output = self.c_attn(x)
    output = self.attn(output)
    output = self.c_attn_proj(output)
    return output

class PositionWiseFF(tf.keras.layers.Layer):
  def __init__(self, n_state: int, d_hid: int, layer_id: int, accurate_gelu: bool):
    super(PositionWiseFF, self).__init__()
    self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
    self.activation = Gelu(accurate=accurate_gelu) #name='layer_{}/gelu'.format(layer_id))
    self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

  def call(self, x, training=True):
    output = self.activation(self.c_fc(x))
    return self.c_ffn_proj(output)

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,
                use_attn_mask: bool, layer_id: int, neg_inf: float, ln_epsilon: float, accurate_gelu: bool):
    super(EncoderLayer, self).__init__()
    self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id, neg_inf)
    self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))
    self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))
    self.ln1 = LayerNormalization(ln_epsilon)# name='layer_{}/ln_1'.format(layer_id))
    self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)
    self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))
    self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))
    self.ln2 = LayerNormalization(ln_epsilon)# name='layer_{}/ln_2'.format(layer_id))

  def call(self, x, training=True):
    a = self.attention(x=x)
    n = self.ln1(self.add1([x, self.drop1(a)]))
    f = self.ffn(n)
    return self.ln2(self.add2([n, self.drop2(f)]))

class Transformer(tf.keras.Model):
  def __init__(self, batch_size, 
              embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30000,
              max_len: int = 512, trainable_pos_embedding: bool = True, num_heads: int = 12,
              num_layers: int = 12, attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,
              d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_attn_mask: bool = False,
              embedding_layer_norm: bool = False, neg_inf: float = -1e9, layer_norm_epsilon: float = 1e-5,
              accurate_gelu: bool = False): 
    super(Transformer,self).__init__()
    self.embedding_dim = embedding_dim
    self.embedding_dropout = embedding_dropout
    self.vocab_size = vocab_size
    self.max_len = max_len
    self.trainable_pos_embedding = trainable_pos_embedding
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.attention_dropout = attention_dropout
    self.use_one_embedding_dropout = use_one_embedding_dropout
    self.d_hid = d_hid 
    self.residual_dropout = residual_dropout
    self.use_attn_mask = use_attn_mask
    self.embedding_layer_norm = embedding_layer_norm
    self.neg_inf = neg_inf
    self.layer_norm_epsilon =layer_norm_epsilon
    self.accurate_gelu =accurate_gelu
    self.pos_emb = keras.layers.Embedding(max_len, embedding_dim, trainable=False, 
      input_length=max_len,name='PositionEmbedding',
      weights=[_get_pos_encoding_matrix(max_len, embedding_dim)])
    self.rv = keras.layers.RepeatVector(max_len)
    self.dense0 = keras.layers.Dense(embedding_dim)
    self.dense1 = keras.layers.TimeDistributed(keras.layers.Dense(vocab_size,activation='relu'))
    self.lstm1 = tf.keras.layers.LSTM(embedding_dim,return_sequences = True,activation ='tanh')

    self.decoder_layers = [EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i, neg_inf, layer_norm_epsilon, accurate_gelu) for i in range(self.num_layers)]
    pos_embed = np.arange(0,max_len)
    batch_pos_embed = []
    for i in range(batch_size):
      batch_pos_embed.append(pos_embed)
    self.batch_pos_embed = np.array(batch_pos_embed)
      
  def call(self, x, training=True):
    pos_embeddings = self.pos_emb(self.batch_pos_embed)
    x = self.rv(x)
    out = self.dense0(x)
    out = keras.layers.Add()([out,pos_embeddings])
    for decoder_layer in self.decoder_layers:
       out = decoder_layer(out)
    out = self.lstm1(out)
    out = self.dense1(out)
    return out