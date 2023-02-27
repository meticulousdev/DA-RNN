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

X_encoded = tf.ones((batch_size, T, m))
print(f"X_encoded: {X_encoded.shape}")

# %%
hidden_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(p)] for _ in range(batch_size)], dtype=tf.float32)
print(f"hidden_state.shape: {hidden_state.shape}")
print(f"cell_state.shape: {cell_state.shape}")

print(f"tf.concat: {tf.concat([hidden_state, cell_state], axis=-1).shape}")

# %%
p = X_encoded.shape[1]
print(f"p: {p}")

concat_ds = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), p)
print(f"concat_ds.shape: {concat_ds.shape}")

ds = Dense(T)(concat_ds)
print(f"ds.shape: {ds.shape}")

uh = Dense(T)(X_encoded)
print(f"uh.shape: {uh.shape}")

add_ds_uh = Add()([ds, uh])
print(f"add_ds_uh.shape: {add_ds_uh.shape}")

tanh_act = Activation(activation='tanh')(add_ds_uh)
print(f"tanh_act.shape: {tanh_act.shape}")

l = Dense(1)(tanh_act)
print(f"l.shape: {l.shape}")

beta = Softmax(axis=1)(l)
print(f"beta.shape: {beta.shape}")

# %%