import random
from typing import List, Optional

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (Dense, 
                                     Permute, 
                                     Softmax, 
                                     Activation,
                                     Add,
                                     Layer,
                                     LSTM)

from tensorflow.keras.models import Model


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
        # alpha_t (batch size, 1, n)
        # X_tilde_t (batch size, 1, n)
        # X_encoded.append(hidden_state[:, None, :]) 
        # (batch size, 1, m) x T
        # encoder_ret (batch size, T, m)

        batch_size = K.shape(X)[0]

        hidden_state = tf.zeros((batch_size, self.m))
        cell_state = tf.zeros((batch_size, self.m))

        X_encoded: List = []         
        for t in range(self.T):
            alpha_t = self.input_attention(hidden_state, cell_state, X)

            # Eqn. (10)
            X_tilde_t = tf.multiply(alpha_t, X[:, None, t, :])

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
        # alpha_add (batch size, 1, n)

        # Eqn. (8)
        n = X.shape[2]

        concat_hs = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), n)
        hs = Dense(self.T)(concat_hs)
        ux = Dense(self.T)(Permute((2, 1))(X))

        add_hs_ux = Add()([hs, ux])
        tanh_act = Activation(activation='tanh')(add_hs_ux)
        e = Dense(1)(tanh_act)

        # Eqn. (9)
        alpha = Softmax()(Permute((2, 1))(e))
        # print(alpha)
        return alpha


class Decoder(Layer):
    def __init__(self, T: int, m: int, p: int, y_dim: int):
        super().__init__()

        self.T = T
        self.m = m
        self.p = p
        self.y_dim = y_dim

        self.decoder_lstm = LSTM(self.p, return_state=True)
    
    def __call__(self, X_encoded, Y):
        # beta_t (batch size, T, 1)
        # c_t (batch size, 1, m)
        # yc_concat (batch size, 1, y_dim + m)
        # y_tilde (batch size, 1, 1)
        # dc_concat (batch size, 1, m + p)
        # y_hat_T (batch size, 1, y_dim)

        batch_size = K.shape(X_encoded)[0]
        hidden_state = tf.zeros((batch_size, self.p))
        cell_state = tf.zeros((batch_size, self.p))

        for t in range(self.T - 1):
            # Eqn. (14)
            beta_t = self.temperal_attention(hidden_state, cell_state, X_encoded)
            c_t = tf.matmul(beta_t, X_encoded, transpose_a=True)

            # Eqn. (15)
            yc_concat = tf.concat([Y[:, None, t, :], c_t], axis=-1)
            y_tilde = Dense(1)(yc_concat)

            # Eqn. (16) (Eqn. (17) - (21))
            hidden_state, _, cell_state = self.decoder_lstm(y_tilde, initial_state=[hidden_state, cell_state])

        # Eqn. (22)
        dc_concat = tf.concat([hidden_state[:, None, :], c_t], axis=-1)
        y_hat_T = Dense(self.y_dim)(Dense(self.p)(dc_concat))
        y_hat_T = tf.squeeze(y_hat_T, axis=1)
        return y_hat_T

    def temperal_attention(self, hidden_state, cell_state, X_encoded):
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

        p = X_encoded.shape[1]
        
        # Eqn. (12)
        concat_ds = K.repeat(tf.concat([hidden_state, cell_state], axis=-1), p)

        ds = Dense(self.m)(concat_ds)
        uh = Dense(self.m)(X_encoded)

        add_ds_uh = Add()([ds, uh])

        tanh_act = Activation(activation='tanh')(add_ds_uh)

        l = Dense(1)(tanh_act)

        # Eqn. (13)
        beta = Softmax(axis=1)(l)
        return beta
    

class DARNN(Model):
    def __init__(self, T: int, m: int, p: Optional[int] = None, y_dim: int = 1) -> None:
        super().__init__()
        
        self.T = T
        self.m = m
        self.p = p or m
        self.y_dim = y_dim
        
        self.encoder = Encoder(self.T, self.m)
        self.decoder = Decoder(self.T, self.m, self.p, self.y_dim)
    
    def __call__ (self, X, Y):
        # X (batch size, T, n)
        # Y (batch size, T-1, y_dim)
        # 
        # X_encoded (batch size, T, m)
        # y_hat_T (batch size, 1, y_dim)
        # Eqn. (1)
        X_encoded = self.encoder(X)
        y_hat_T = self.decoder(Y, X_encoded)

        return y_hat_T
    

def test_encoder_merged_class(batch_size: int, T: int, n: int, m: int):
    # ret_encoder = test_encoder_merged_class(batch_size, T, n, m)
    # print(ret_encoder)
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


def test_decoder_merged_class(batch_size: int, T: int, m: int, p: int, y_dim: int):
    # ret_decoder = test_decoder_merged_class(batch_size, T, m, p, y_dim)
    # print(ret_decoder)
    random.seed(42)

    print(tf.__version__)
    tf.random.set_seed(42)

    X_encoded = tf.random.uniform((batch_size, T, m))

    Y = tf.random.uniform((batch_size, T - 1, y_dim))
    da_rnn_decoder = Decoder(T, m, p, y_dim)
    ret = da_rnn_decoder(X_encoded, Y)
    return ret


def test_da_rnn_merged_class(batch_size: int, T: int, m: int, p: int, y_dim: int):
    # ret_da_rnn = test_da_rnn_merged_class(batch_size, T, m, p, y_dim)
    # print(ret_da_rnn)
    random.seed(42)

    print(tf.__version__)
    tf.random.set_seed(42)

    inputs = tf.random.uniform((batch_size, T, m + y_dim))
    # X (batch size, T, n)       : x(1), x(2), ..., x(T)
    # Y (batch size, T-1, y_dim) : y(1), y(2), ..., y(T-1)
    X = inputs[:, :, :-y_dim]
    Y = inputs[:, :-1, -y_dim:]

    da_rnn = DARNN(T, m, p, y_dim)
    ret = da_rnn(X, Y)
    return ret


if __name__ == "__main__":
    random.seed(42)
    tf.random.set_seed(42)

    batch_size = 12
    T = 5
    n = 4
    p = 6
    m = 7

    y_dim = 3

    da_rnn = DARNN(T, m, p, y_dim)
    da_rnn.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    # da_rnn.summary()

    # 2023-04-05 18:42:29.738490: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with 
    # oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
    # To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    # 2023-04-05 18:42:30.162722: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6005 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2060 SUPER, pci bus id: 0000:08:00.0, 
    # compute capability: 7.5
    