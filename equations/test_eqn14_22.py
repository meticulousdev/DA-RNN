# %%
import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, LSTM

from test_model_separated_class import TemperalAttention

# %%
random.seed(42)

print(tf.__version__)
tf.random.set_seed(42)

# %%
batch_size = 7
T = 5
p = 4
m = 3

y_dim = 3

hidden_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)

X_encoded = tf.ones((batch_size, T, m))

# TODO T or T - 1
Y = tf.ones((batch_size, T - 1, y_dim))

# %%
temperal_attention = TemperalAttention(m)
decoder_lstm = LSTM(p, return_state=True)

t = 1
h_encoded = []
# %%
# beta_t (batch size, T, 1)
# c_t (batch size, 1, m)
# yc_concat (batch size, 1, y_dim + m)
# y_tilde (batch size, 1, 1)

beta_t = temperal_attention(hidden_state, cell_state, X_encoded)

# TODO transpose_a
# Eqn. (14)
c_t = tf.matmul(beta_t, X_encoded, transpose_a=True)

# Eqn. (15)
yc_concat = tf.concat([Y[:, None, t, :], c_t], axis=-1)
y_tilde = Dense(1)(yc_concat)

# %%
print(f"beta_t.shape: {beta_t.shape}")
print(f"c_t.shape: {c_t.shape}")
print(f"yc_concat.shape: {yc_concat.shape}")
print(f"y_tilde.shape: {y_tilde.shape}")

# %%
# temperal_attention = TemperalAttention(m)
# for t in range(T):
#     beta_t = temperal_attention(hidden_state, cell_state, X_encoded)

#     c_t = tf.matmul(beta_t, X_encoded)
