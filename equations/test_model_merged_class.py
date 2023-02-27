import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Dense, 
    Permute, 
    Softmax, 
    Activation, 
    Add,
    Layer,
    LSTM
)


# TODO super().__init__(name='input_attention')
#      super().__init__(name='encoder_input')
# TODO get_config
class Encoder(Layer):
    def __init__(self, T: int, m: int) -> None:
        super().__init__()

        self.T = T
        self.m = m
        
        self.input_lstm = LSTM(m, return_state=True)

    def __call__(self, X):
        # X (batch size, T, n) 
        # 
        # hidden_state (batch size, m)
        # cell_state (batch size, m)
        #
        # attn_t (batch size, 1, n)
        # X_tilde_t (batch size, 1, n)
        # X_encoded.append(hidden_state[:, None, :]) 
        # (batch size, 1, m) x T
        # encoder_ret (batch size, T, m)

        batch_size = K.shape(X)[0]

        hidden_state = tf.zeros((batch_size, self.m))
        cell_state = tf.zeros((batch_size, self.m))

        X_encoded = []        
        for t in range(self.T):
            attn_t = self.input_attention(hidden_state, cell_state, X)

            # Eqn. (10)
            X_tilde_t = tf.multiply(attn_t, X[:, None, t, :])

            # Eqn. (11)
            hidden_state, _, cell_state = self.input_lstm(X_tilde_t, initial_state=[hidden_state, cell_state])

            X_encoded.append(hidden_state[:, None, :])
        
        encoder_ret = tf.concat(X_encoded, axis=1)
        return encoder_ret


    def input_attention(self, hidden_state, cell_state, X):
        # hidden_state (batch size, m)
        # cell_state (batch size, m)    
        # X (batch size, T, n)
        #
        # concat_hs (batch size, n, 2m)
        # hs (batch size, n, T)
        # ux (batch size, n, T)
        #
        # add_hs_ux (batch size, n, T)
        # tanh_act_add (batch size, n, T)
        # e_add (batch size, n, 1)
        #
        # attn_add (batch size, 1, n)

        # Eqn. (8)
        n = X.shape[2]

        concat_hs = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), n)
        hs = Dense(self.T)(concat_hs)
        ux = Dense(self.T)(Permute((2, 1))(X))

        add_hs_ux = Add()([hs, ux])
        tanh_act = Activation(activation='tanh')(add_hs_ux)
        e = Dense(1)(tanh_act)

        # Eqn. (9)
        attn = Softmax()(Permute((2, 1))(e))
        # print(attn)
        return attn


def test_encoder_merged_class(batch_size: int, T: int, n: int, m: int):
    random.seed(42)

    print(tf.__version__)
    tf.random.set_seed(42)
        
    ele01 = [random.random() for _ in range(n)]
    ele02 = [ele01 for _ in range(T)]
    ele03 = [ele02 for _ in range(batch_size)]

    X = tf.constant(ele03, dtype=tf.float32)

    da_rnn_encoder = Encoder(T, m)
    ret = da_rnn_encoder(X)
    return ret


if __name__ == "__main__":
    batch_size = 7
    T = 5
    n = 4
    m = 3

    ret = test_encoder_merged_class(batch_size, T, n, m)
    print(ret)
