from test_model_merged_class import test_encoder_merged_class
from test_model_separated_class import test_encoder_seprated_class

if __name__ == "__main__":
    batch_size = 7
    T = 5
    n = 4
    m = 3

    ret_sep = test_encoder_seprated_class(batch_size, T, n, m)
    ret_mer = test_encoder_merged_class(batch_size, T, n, m)

    print(sum(ret_sep - ret_mer))
