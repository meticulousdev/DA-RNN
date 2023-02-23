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

class InputAttention(Layer):
    def __init__(self) -> None:
        # TODO super.__init__(name=???)
        super.__init__(name='input_attention')

    # TODO call
    # TODO return type
    def call(self, hidden_state, cell_state, X):
        # X (batch size, T, n)
        #
        # concat_hs (batach, n, 2m)
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
        hs = Dense(T)(concat_hs)
        ux = Dense(T)(Permute((2, 1))(X))

        add_hs_ux = Add()([hs, ux])
        tanh_act = Activation(activation='tanh')(add_hs_ux)
        e = Dense(1)(tanh_act)

        # Eqn. (9)
        attn = Softmax()(Permute((2, 1))(e))
        # print(attn)
        return attn
    
    # TODO get_config ???
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'T': self.T
    #     })
    #     return config


class Encoder(Layer):
    def __init__(self) -> None:
        super().__init__()


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
