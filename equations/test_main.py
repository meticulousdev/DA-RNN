from test_model_merged_class import test_encoder_merged_class, test_decoder_merged_class
from test_model_separated_class import test_encoder_separated_class, test_decoder_separated_class

if __name__ == "__main__":
    batch_size = 12
    T = 5
    n = 4
    p = 6
    m = 4

    y_dim = 3

    ret_en_sep = test_encoder_separated_class(batch_size, T, n, m)
    ret_en_mer = test_encoder_merged_class(batch_size, T, n, m)
    print(sum(ret_en_sep - ret_en_mer))

    ret_de_sep = test_decoder_separated_class(batch_size, T, m, p, y_dim)
    ret_de_mer = test_decoder_merged_class(batch_size, T, m, p, y_dim)
    print(sum(ret_de_sep - ret_de_mer))

    # tf.Tensor(
    # [[0. 0. 0. 0.]
    # [0. 0. 0. 0.]
    # [0. 0. 0. 0.]
    # [0. 0. 0. 0.]
    # [0. 0. 0. 0.]], shape=(5, 4), dtype=float32)

    # tf.Tensor([0. 0. 0.], shape=(3,), dtype=float32)
