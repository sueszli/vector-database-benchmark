import numpy as np

def fake_imdb_reader(word_dict_size, sample_num, lower_seq_len=100, upper_seq_len=200, class_dim=2):
    if False:
        return 10

    def __reader__():
        if False:
            i = 10
            return i + 15
        for _ in range(sample_num):
            length = np.random.random_integers(low=lower_seq_len, high=upper_seq_len, size=[1])[0]
            ids = np.random.random_integers(low=0, high=word_dict_size - 1, size=[length]).astype('int64')
            label = np.random.random_integers(low=0, high=class_dim - 1, size=[1]).astype('int64')[0]
            yield (ids, label)
    return __reader__