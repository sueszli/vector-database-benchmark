"""
Defines text datatset preprocessing routines
"""
from future import standard_library
standard_library.install_aliases()
from builtins import map
import numpy as np
import re
from neon.util.compat import pickle

def clean_string(base):
    if False:
        while True:
            i = 10
    '\n    Tokenization/string cleaning.\n    Original from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py\n    '
    base = re.sub("[^A-Za-z0-9(),!?\\'\\`]", ' ', base)
    base = re.sub("\\'re", " 're", base)
    base = re.sub("\\'d", " 'd", base)
    base = re.sub("\\'ll", " 'll", base)
    base = re.sub("\\'s", " 's", base)
    base = re.sub("\\'ve", " 've", base)
    base = re.sub("n\\'t", " n't", base)
    base = re.sub('!', ' ! ', base)
    base = re.sub(',', ' , ', base)
    base = re.sub('\\)', ' \\) ', base)
    base = re.sub('\\(', ' \\( ', base)
    base = re.sub('\\?', ' \\? ', base)
    base = re.sub('\\s{2,}', ' ', base)
    return base.strip().lower()

def pad_sentences(sentences, sentence_length=None, dtype=np.int32, pad_val=0.0):
    if False:
        for i in range(10):
            print('nop')
    lengths = [len(sent) for sent in sentences]
    nsamples = len(sentences)
    if sentence_length is None:
        sentence_length = np.max(lengths)
    X = (np.ones((nsamples, sentence_length)) * pad_val).astype(dtype=np.int32)
    for (i, sent) in enumerate(sentences):
        trunc = sent[-sentence_length:]
        X[i, -len(trunc):] = trunc
    return X

def pad_data(path, vocab_size=20000, sentence_length=100, oov=2, start=1, index_from=3, seed=113, test_split=0.2):
    if False:
        for i in range(10):
            print('nop')
    f = open(path, 'rb')
    (X, y) = pickle.load(f)
    f.close()
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]
    if not vocab_size:
        vocab_size = max([max(x) for x in X])
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]
    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = y[:int(len(X) * (1 - test_split))]
    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = y[int(len(X) * (1 - test_split)):]
    X_train = pad_sentences(X_train, sentence_length=sentence_length)
    y_train = np.array(y_train).reshape((len(y_train), 1))
    X_test = pad_sentences(X_test, sentence_length=sentence_length)
    y_test = np.array(y_test).reshape((len(y_test), 1))
    nclass = 1 + max(np.max(y_train), np.max(y_test))
    return ((X_train, y_train), (X_test, y_test), nclass)

def get_paddedXY(X, y, vocab_size=20000, sentence_length=100, oov=2, start=1, index_from=3, seed=113, shuffle=True):
    if False:
        i = 10
        return i + 15
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]
    if not vocab_size:
        vocab_size = max([max(x) for x in X])
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]
    else:
        X = [[w for w in x if w < vocab_size] for x in X]
    X = pad_sentences(X, sentence_length=sentence_length)
    y = np.array(y, dtype=np.int32).reshape((len(y), 1))
    return (X, y)

def get_google_word2vec_W(fname, vocab, vocab_size=1000000, index_from=3):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract the embedding matrix from the given word2vec binary file and use this\n    to initalize a new embedding matrix for words found in vocab.\n\n    Conventions are to save indices for pad, oov, etc.:\n    index 0: pad\n    index 1: oov (or <unk>)\n    index 2: <eos>. But often cases, the <eos> has already been in the\n    preprocessed data, so no need to save an index for <eos>\n    '
    f = open(fname, 'rb')
    header = f.readline()
    (vocab1_size, embedding_dim) = list(map(int, header.split()))
    binary_len = np.dtype('float32').itemsize * embedding_dim
    vocab_size = min(len(vocab) + index_from, vocab_size)
    W = np.zeros((vocab_size, embedding_dim))
    found_words = {}
    for (i, line) in enumerate(range(vocab1_size)):
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = b''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if word in vocab:
            wrd_id = vocab[word] + index_from
            if wrd_id < vocab_size:
                W[wrd_id] = np.fromstring(f.read(binary_len), dtype='float32')
                found_words[wrd_id] = 1
        else:
            f.read(binary_len)
    cnt = 0
    for wrd_id in range(vocab_size):
        if wrd_id not in found_words:
            W[wrd_id] = np.random.uniform(-0.25, 0.25, embedding_dim)
            cnt += 1
    assert cnt + len(found_words) == vocab_size
    f.close()
    return (W, embedding_dim, vocab_size)