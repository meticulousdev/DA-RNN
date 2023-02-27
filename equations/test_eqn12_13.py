# %%
# temperal_attention
import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Permute, Softmax, Activation, Add

# %%
random.seed(42)

print(tf.__version__)
tf.random.set_seed(42)

# %%
#### tf.keras.layers.Dense
#### tf.keras.layers.Permute
batch_size = 7
T = 5
p = 4
m = 3

# X_encoded (batch size, T, m)
# hidden_state (T, p)
# cell_state (T, p)
#
# tf.concat (batch size, 2p)
# concat_ds (batch size, T, 2p)
# ds (batch size, T, m)
# uh (batch size, T, m)
# 
# add_ds_uh (batch size, T, m)
# tanh_act (batch size, T, m)
# l (batch size, T, 1)
#
# beta (batch size, T, 1)

X_encoded = tf.ones((batch_size, T, m))

# %%
hidden_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)

# %%

p = X_encoded.shape[1]

concat_ds = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), p)

ds = Dense(m)(concat_ds)
uh = Dense(m)(X_encoded)

add_ds_uh = Add()([ds, uh])

tanh_act = Activation(activation='tanh')(add_ds_uh)

l = Dense(1)(tanh_act)

beta = Softmax(axis=1)(l)

# %%
print(f"X_encoded: {X_encoded.shape}")

print(f"hidden_state.shape: {hidden_state.shape}")
print(f"cell_state.shape: {cell_state.shape}")

print(f"tf.concat: {tf.concat([hidden_state, cell_state], axis=-1).shape}")

print(f"concat_ds.shape: {concat_ds.shape}")

print(f"ds.shape: {ds.shape}")
print(f"uh.shape: {uh.shape}")

print(f"add_ds_uh.shape: {add_ds_uh.shape}")
print(f"tanh_act.shape: {tanh_act.shape}")

print(f"l.shape: {l.shape}")

print(f"beta.shape: {beta.shape}")

# %%
# X_encoded: (7, 5, 3)
# hidden_state.shape: (7, 4)
# cell_state.shape: (7, 4)

# tf.concat: (7, 8)
# concat_ds.shape: (7, 5, 8)
# ds.shape: (7, 5, 3)
# uh.shape: (7, 5, 3)

# add_ds_uh.shape: (7, 5, 3)
# tanh_act.shape: (7, 5, 3)
# l.shape: (7, 5, 1)

# beta.shape: (7, 5, 1)