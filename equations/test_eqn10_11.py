# %%
from test_model import *

from tensorflow.keras.layers import LSTM

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
t = 1
X_tilde_t = tf.multiply(attn, X[:, None, t, :])
print(X_tilde_t.shape)

# %%
hidden_state, _, cell_state = LSTM(m, return_state=True)(X_tilde_t, initial_state=[hidden_state, cell_state])
X_encoded.append(hidden_state[:, None, :])

# %%
tf.concat(X_encoded, axis=1)

