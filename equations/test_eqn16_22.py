# %%
import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Permute, Softmax, Activation, Add

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

hidden_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)

X_encoded = tf.ones((batch_size, T, m))

# %%
h_encoded = []
temperal_attention = TemperalAttention(m)

beta_t = temperal_attention(hidden_state, cell_state, X_encoded)

c_t = tf.matmul(beta_t, X_encoded)

# %%
# temperal_attention = TemperalAttention(m)
# for t in range(T):
#     beta_t = temperal_attention(hidden_state, cell_state, X_encoded)

#     c_t = tf.matmul(beta_t, X_encoded)
