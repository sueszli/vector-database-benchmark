import os
import numpy
from chainer.dataset import download

def get_ptb_words():
    if False:
        i = 10
        return i + 15
    "Gets the Penn Tree Bank dataset as long word sequences.\n\n    `Penn Tree Bank <https://catalog.ldc.upenn.edu/LDC99T42>`_\n    is originally a corpus of English sentences with linguistic structure\n    annotations. This function uses a variant distributed at\n    `https://github.com/wojzaremba/lstm <https://github.com/wojzaremba/lstm>`_,\n    which omits the annotation and splits the dataset into three parts:\n    training, validation, and test.\n\n    This function returns the training, validation, and test sets, each of\n    which is represented as a long array of word IDs. All sentences in the\n    dataset are concatenated by End-of-Sentence mark '<eos>', which is treated\n    as one of the vocabulary.\n\n    Returns:\n        tuple of numpy.ndarray: Int32 vectors of word IDs.\n\n    .. Seealso::\n       Use :func:`get_ptb_words_vocabulary` to get the mapping between the\n       words and word IDs.\n\n    "
    train = _retrieve_ptb_words('train.npz', _train_url)
    valid = _retrieve_ptb_words('valid.npz', _valid_url)
    test = _retrieve_ptb_words('test.npz', _test_url)
    return (train, valid, test)

def get_ptb_words_vocabulary():
    if False:
        i = 10
        return i + 15
    'Gets the Penn Tree Bank word vocabulary.\n\n    Returns:\n        dict: Dictionary that maps words to corresponding word IDs. The IDs are\n        used in the Penn Tree Bank long sequence datasets.\n\n    .. seealso::\n       See :func:`get_ptb_words` for the actual datasets.\n\n    '
    return _retrieve_word_vocabulary()
_train_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
_valid_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'
_test_url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'

def _retrieve_ptb_words(name, url):
    if False:
        return 10

    def creator(path):
        if False:
            return 10
        vocab = _retrieve_word_vocabulary()
        words = _load_words(url)
        x = numpy.empty(len(words), dtype=numpy.int32)
        for (i, word) in enumerate(words):
            x[i] = vocab[word]
        numpy.savez_compressed(path, x=x)
        return {'x': x}
    root = download.get_dataset_directory('pfnet/chainer/ptb')
    path = os.path.join(root, name)
    loaded = download.cache_or_load_file(path, creator, numpy.load)
    return loaded['x']

def _retrieve_word_vocabulary():
    if False:
        while True:
            i = 10

    def creator(path):
        if False:
            i = 10
            return i + 15
        words = _load_words(_train_url)
        vocab = {}
        index = 0
        with open(path, 'w') as f:
            for word in words:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                    f.write(word + '\n')
        return vocab

    def loader(path):
        if False:
            i = 10
            return i + 15
        vocab = {}
        with open(path) as f:
            for (i, word) in enumerate(f):
                vocab[word.strip()] = i
        return vocab
    root = download.get_dataset_directory('pfnet/chainer/ptb')
    path = os.path.join(root, 'vocab.txt')
    return download.cache_or_load_file(path, creator, loader)

def _load_words(url):
    if False:
        for i in range(10):
            print('nop')
    path = download.cached_download(url)
    words = []
    with open(path) as words_file:
        for line in words_file:
            if line:
                words += line.strip().split()
                words.append('<eos>')
    return words