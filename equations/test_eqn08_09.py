# %%
# input_attention
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

ele01 = [random.random() for _ in range(4)]
ele02 = [ele01 for _ in range(5)]
ele03 = [ele02 for _ in range(7)]

X = tf.constant(ele03, dtype=tf.float32)

# print(X)
print(X.shape)
print()

X_tr = Permute((2, 1))(X)
# print(X_tr)
print(X_tr.shape)

# %%
hidden_state = tf.constant([[random.random() for _ in range(3)] for _ in range(7)], dtype=tf.float32)
cell_state = tf.constant([[random.random() for _ in range(3)] for _ in range(7)], dtype=tf.float32)

print(tf.concat([hidden_state, cell_state], axis=-1).shape)

# %%
n = X.shape[2]

concat_hs = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), n)
print(concat_hs.shape)

# %%
T = 6
hs = Dense(T)(concat_hs)
print(Dense(T)(hs).shape)

# %%
ux = Dense(T)(Permute((2, 1))(X))
print(ux.shape)

# %%
print(hs[0, 0, :])

print(ux[0, 0, :])

# %%
### tf.keras.layers.Add
#### tf.math.tanh
#### tf.keras.layers.Activation
temp_add = Add()([hs, ux])

print(temp_add.shape)

print(temp_add[0, 0, :])

# %%
tanh_math_add = tf.math.tanh(temp_add)
print(tanh_math_add.shape)

# %%
tanh_act_add = Activation(activation='tanh')(temp_add)
print(tanh_act_add.shape)

# %%
diff_tanh_add = tanh_math_add - tanh_act_add
print(sum(sum(sum(diff_tanh_add))))

# %%
e_add = Dense(1)(tanh_act_add)
print(e_add[:, :, 0])
print(e_add.shape)

# %%
attn_add = Softmax()(e_add)
print(attn_add[:, :, 0])
print(attn_add.shape)

# %%
attn_add = Softmax()(Permute((2, 1))(e_add))
print(attn_add[:, 0, :])
print(attn_add.shape)

# %%
### tf.concat
#### tf.math.tanh
#### tf.keras.layers.Activation
temp_concat = tf.concat([hs, ux], axis=-1)

print(temp_concat.shape)

print(temp_concat[0, 0, :])

# %%
tanh_math_concat = tf.math.tanh(temp_concat)
print(tanh_math_concat.shape)

# %%
tanh_act_concat = Activation(activation='tanh')(temp_concat)
print(tanh_act_concat.shape)

# %%
diff_tanh = tanh_math_concat - tanh_act_concat
print(sum(sum(sum(diff_tanh))))

# %%
e_act = Dense(1)(tanh_act_concat)
print(e_act[:, :, 0])
print(e_act.shape)

# %%
attn_input_act = Softmax()(e_act)
print(attn_input_act[:, :, 0])
print(attn_input_act.shape)

# %%
attn_input_act = Softmax()(Permute((2, 1))(e_act))
print(attn_input_act[:, 0, :])
print(attn_input_act.shape)
