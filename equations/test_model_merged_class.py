import random

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Dense, 
    Permute, 
    Softmax, 
    Activation, 
    Add,
    Layer
)


# TODO super().__init__(name='input_attention')
#      super().__init__(name='encoder_input')
# TODO get_config
class Encoder(Layer):
    def __init__(self, T: int, m: int) -> None:
        super().__init__()

        self.T = T
        self.m = m

    def __call__(self, X):
        batch_size = K.shape(X)[0]

        hidden_state = tf.zeros((batch_size, self.m))
        cell_state = tf.zeros((batch_size, self.m))

        X_encoded = []        
        for t in range(self.T):
            attn_t = self.input_attention(hidden_state, cell_state, X)

    def input_encoder(self):
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


if __name__ == "__main__":
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
