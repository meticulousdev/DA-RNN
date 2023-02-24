# %%
from test_model import *
import random

from tensorflow.keras.layers import LSTM


# %%
batch_size = 7
T = 5
n = 4
m = 3

ele01 = [random.random() for _ in range(n)]
ele02 = [ele01 for _ in range(T)]
ele03 = [ele02 for _ in range(batch_size)]

X = tf.constant(ele03, dtype=tf.float32)

hs_ele01 = [random.random() for _ in range(m)]
hs_ele02 = [hs_ele01 for _ in range(batch_size)]
hidden_state = tf.constant(hs_ele02, dtype=tf.float32)

cs_ele01 = [random.random() for _ in range(m)]
cs_ele02 = [cs_ele01 for _ in range(batch_size)]
cell_state = tf.constant(cs_ele02, dtype=tf.float32)

# %%
# TODO
# self.input_lstm = LSTM(m, return_state=True)
# self.input_attention = InputAttention(T)
#
# hidden_state와 cell_state의 업데이트를 반영할 수 있는 코드

# %%
X_encoded = []

# %%
for t in range(T):
    temp = X[:, None, t, :]
    print(temp.shape)

# %%
# TODO: multiply and Dense?
# DONE: Dense - parameters to learn
input_attention = InputAttention(T)
attn = input_attention(hidden_state, cell_state, X)

t = 1
# Eqn. (10)
X_tilde_t = tf.multiply(attn, X[:, None, t, :])
print(X_tilde_t.shape)

# %%
# Eqn. (11)
hidden_state, _, cell_state = LSTM(m, return_state=True)(X_tilde_t, initial_state=[hidden_state, cell_state])
X_encoded.append(hidden_state[:, None, :])

# %%
tf.concat(X_encoded, axis=1)
