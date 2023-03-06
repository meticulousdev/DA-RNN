# %%
import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, LSTM, Permute

from test_model_separated_class import TemperalAttention

# %%
random.seed(42)

print(tf.__version__)
tf.random.set_seed(42)

# %%
batch_size = 12
T = 5
p = 6
m = 4

y_dim = 3

hidden_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)

X_encoded = tf.ones((batch_size, T, m))

Y = tf.ones((batch_size, T - 1, y_dim))

# %%
temperal_attention = TemperalAttention(m)
decoder_lstm = LSTM(p, return_state=True)

t = 1
h_encoded = []
# %%
beta_t = temperal_attention(hidden_state, cell_state, X_encoded)

print(f"beta_t.shape: {beta_t.shape}")
print(f"X_encoded.shape: {X_encoded.shape}")
print(f"Permute((2, 1))(beta_t).shape: {Permute((2, 1))(beta_t).shape}")
print(f"tf.transpose(beta_t.shape): {tf.transpose(beta_t).shape}\n")

# %%
# TODO transpose_a
# transpose_a, transpose_b
# Eqn. (14)
beta_t = temperal_attention(hidden_state, cell_state, X_encoded)
c_t = tf.matmul(beta_t, X_encoded, transpose_a=True)

# beta_t = temperal_attention(hidden_state, cell_state, X_encoded)
# c_t_p = tf.matmul(Permute((2, 1))(beta_t), X_encoded)

# print(c_t)
# print(c_t_p)
# %%
# Eqn. (15)
yc_concat = tf.concat([Y[:, None, t, :], c_t], axis=-1)
y_tilde = Dense(1)(yc_concat)

# %%
print(f"beta_t.shape: {beta_t.shape}")
print(f"c_t.shape: {c_t.shape}")
print(f"yc_concat.shape: {yc_concat.shape}")
print(f"y_tilde.shape: {y_tilde.shape}\n")

# %%
# beta_t (batch size, T, 1)
# c_t (batch size, 1, m)
# yc_concat (batch size, 1, y_dim + m)
# y_tilde (batch size, 1, 1)
# dc_concat (batch size, 1, m + p)
# y_hat_T (batch size, 1, y_dim)

print("for loop")
temperal_attention = TemperalAttention(m)
for t in range(T - 1):
    # Eqn. (14)
    beta_t = temperal_attention(hidden_state, cell_state, X_encoded)
    c_t = tf.matmul(beta_t, X_encoded, transpose_a=True)

    # Eqn. (15)
    yc_concat = tf.concat([Y[:, None, t, :], c_t], axis=-1)
    y_tilde = Dense(1)(yc_concat)

    # Eqn. (16) (Eqn. (17) - (21))
    hidden_state, _, cell_state = decoder_lstm(y_tilde, initial_state=[hidden_state, cell_state])

# Eqn. (22)
dc_concat = tf.concat([hidden_state[:, None, :], c_t], axis=-1)
y_hat_T = Dense(y_dim)(Dense(p)(dc_concat))

print(f"beta_t.shape: {beta_t.shape}")
print(f"c_t.shape: {c_t.shape}")
print(f"yc_concat.shape: {yc_concat.shape}")
print(f"y_tilde.shape: {y_tilde.shape}")
print(f"dc_concat.shape: {dc_concat.shape}")
print(f"y_hat_T.shape: {y_hat_T.shape}\n")

# for loop
# beta_t.shape: (12, 5, 1)
# c_t.shape: (12, 1, 4)
# yc_concat.shape: (12, 1, 7)
# y_tilde.shape: (12, 1, 1)
# dc_concat.shape: (12, 1, 10)
# y_hat_T.shape: (12, 1, 3)
