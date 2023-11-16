import os
from tensorlayer import logging, nlp
from tensorlayer.files.utils import maybe_download_and_extract
__all__ = ['load_ptb_dataset']

def load_ptb_dataset(path='data'):
    if False:
        for i in range(10):
            print('nop')
    'Load Penn TreeBank (PTB) dataset.\n\n    It is used in many LANGUAGE MODELING papers,\n    including "Empirical Evaluation and Combination of Advanced Language\n    Modeling Techniques", "Recurrent Neural Network Regularization".\n    It consists of 929k training words, 73k validation words, and 82k test\n    words. It has 10k words in its vocabulary.\n\n    Parameters\n    ----------\n    path : str\n        The path that the data is downloaded to, defaults is ``data/ptb/``.\n\n    Returns\n    --------\n    train_data, valid_data, test_data : list of int\n        The training, validating and testing data in integer format.\n    vocab_size : int\n        The vocabulary size.\n\n    Examples\n    --------\n    >>> train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()\n\n    References\n    ---------------\n    - ``tensorflow.models.rnn.ptb import reader``\n    - `Manual download <http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz>`__\n\n    Notes\n    ------\n    - If you want to get the raw data, see the source code.\n\n    '
    path = os.path.join(path, 'ptb')
    logging.info('Load or Download Penn TreeBank (PTB) dataset > {}'.format(path))
    filename = 'simple-examples.tgz'
    url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
    maybe_download_and_extract(filename, path, url, extract=True)
    data_path = os.path.join(path, 'simple-examples', 'data')
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')
    word_to_id = nlp.build_vocab(nlp.read_words(train_path))
    train_data = nlp.words_to_word_ids(nlp.read_words(train_path), word_to_id)
    valid_data = nlp.words_to_word_ids(nlp.read_words(valid_path), word_to_id)
    test_data = nlp.words_to_word_ids(nlp.read_words(test_path), word_to_id)
    vocab_size = len(word_to_id)
    return (train_data, valid_data, test_data, vocab_size)