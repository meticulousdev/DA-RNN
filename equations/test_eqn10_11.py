# %%
from test_model_separated_class import *
import random

from tensorflow.keras.layers import LSTM


# %%
random.seed(42)

print(tf.__version__)
tf.random.set_seed(42)

# %%
batch_size = 7
T = 5
n = 4
m = 3

ele01 = [random.random() for _ in range(n)]
ele02 = [ele01 for _ in range(T)]
ele03 = [ele02 for _ in range(batch_size)]

X = tf.constant(ele03, dtype=tf.float32)

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
hidden_state = tf.zeros((batch_size, m))
cell_state = tf.zeros((batch_size, m))

input_attention = InputAttention(T)
input_lstm = LSTM(m, return_state=True)

for t in range(T):
    attn_t = input_attention(hidden_state, cell_state, X)

    # Eqn. (10)
    X_tilde_t = tf.multiply(attn_t, X[:, None, t, :])

    # Eqn. (11)

    hidden_state, _, cell_state = input_lstm(X_tilde_t, initial_state=[hidden_state, cell_state])

    X_encoded.append(hidden_state[:, None, :])
    print(f"{id(input_lstm)} {id(input_attention)}")    
    print(f"{t} {attn_t.shape} {X_tilde_t.shape} {hidden_state.shape}\n")

# %%
# print(X_encoded)
ret = tf.concat(X_encoded, axis=1)
print(ret.shape)
