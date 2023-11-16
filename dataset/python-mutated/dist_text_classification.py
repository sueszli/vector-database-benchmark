import os
import re
import string
import tarfile
import nets
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle
from paddle import base
DTYPE = 'float32'
VOCAB_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/imdb.vocab'
VOCAB_MD5 = '23c86a0533c0151b6f12fa52b106dcc2'
DATA_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/text_classification.tar.gz'
DATA_MD5 = '29ebfc94f11aea9362bbb7f5e9d86b8a'

def load_vocab(filename):
    if False:
        while True:
            i = 10
    vocab = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for (idx, line) in enumerate(f):
            vocab[line.strip()] = idx
    return vocab

def get_worddict(dict_path):
    if False:
        i = 10
        return i + 15
    word_dict = load_vocab(dict_path)
    word_dict['<unk>'] = len(word_dict)
    dict_dim = len(word_dict)
    return (word_dict, dict_dim)

def conv_net(input, dict_dim, emb_dim=128, window_size=3, num_filters=128, fc0_dim=96, class_dim=2):
    if False:
        print('Hello World!')
    emb = paddle.static.nn.embedding(input=input, size=[dict_dim, emb_dim], is_sparse=False, param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    conv_3 = nets.sequence_conv_pool(input=emb, num_filters=num_filters, filter_size=window_size, act='tanh', pool_type='max', param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    fc_0 = paddle.static.nn.fc(x=[conv_3], size=fc0_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    prediction = paddle.static.nn.fc(x=[fc_0], size=class_dim, activation='softmax', weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    return prediction

def inference_network(dict_dim):
    if False:
        for i in range(10):
            print('nop')
    data = paddle.static.data(name='words', shape=[-1, 1], dtype='int64', lod_level=1)
    out = conv_net(data, dict_dim)
    return out

def get_reader(word_dict, batch_size):
    if False:
        return 10
    train_reader = paddle.batch(train(word_dict), batch_size=batch_size)
    test_reader = paddle.batch(test(word_dict), batch_size=batch_size)
    return (train_reader, test_reader)

def get_optimizer(learning_rate):
    if False:
        while True:
            i = 10
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate)
    return optimizer

class TestDistTextClassification2x2(TestDistRunnerBase):

    def get_model(self, batch_size=2):
        if False:
            return 10
        vocab = os.path.join(paddle.dataset.common.DATA_HOME, 'text_classification', 'imdb.vocab')
        (word_dict, dict_dim) = get_worddict(vocab)
        data = paddle.static.data(name='words', shape=[-1, 1], dtype='int64', lod_level=1)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        predict = conv_net(data, dict_dim)
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=predict, label=label)
        inference_program = base.default_main_program().clone()
        opt = get_optimizer(learning_rate=0.001)
        opt.minimize(avg_cost)
        (train_reader, test_reader) = get_reader(word_dict, batch_size)
        return (inference_program, avg_cost, train_reader, test_reader, acc, predict)

def tokenize(pattern):
    if False:
        for i in range(10):
            print('nop')
    '\n    Read files that match the given pattern.  Tokenize and yield each file.\n    '
    with tarfile.open(paddle.dataset.common.download(DATA_URL, 'text_classification', DATA_MD5)) as tarf:
        tf = tarf.next()
        while tf is not None:
            if bool(pattern.match(tf.name)):
                yield tarf.extractfile(tf).read().rstrip(b'\n\r').translate(None, string.punctuation.encode('latin-1')).lower().split()
            tf = tarf.next()

def reader_creator(pos_pattern, neg_pattern, word_idx):
    if False:
        for i in range(10):
            print('nop')
    UNK = word_idx['<unk>']
    INS = []

    def load(pattern, out, label):
        if False:
            return 10
        for doc in tokenize(pattern):
            out.append(([word_idx.get(w, UNK) for w in doc], label))
    load(pos_pattern, INS, 0)
    load(neg_pattern, INS, 1)

    def reader():
        if False:
            return 10
        yield from INS
    return reader

def train(word_idx):
    if False:
        print('Hello World!')
    '\n    IMDB training set creator.\n\n    It returns a reader creator, each sample in the reader is an zero-based ID\n    sequence and label in [0, 1].\n\n    :param word_idx: word dictionary\n    :type word_idx: dict\n    :return: Training reader creator\n    :rtype: callable\n    '
    return reader_creator(re.compile('train/pos/.*\\.txt$'), re.compile('train/neg/.*\\.txt$'), word_idx)

def test(word_idx):
    if False:
        print('Hello World!')
    '\n    IMDB test set creator.\n\n    It returns a reader creator, each sample in the reader is an zero-based ID\n    sequence and label in [0, 1].\n\n    :param word_idx: word dictionary\n    :type word_idx: dict\n    :return: Test reader creator\n    :rtype: callable\n    '
    return reader_creator(re.compile('test/pos/.*\\.txt$'), re.compile('test/neg/.*\\.txt$'), word_idx)
if __name__ == '__main__':
    paddle.dataset.common.download(VOCAB_URL, 'text_classification', VOCAB_MD5)
    paddle.dataset.common.download(DATA_URL, 'text_classification', DATA_MD5)
    runtime_main(TestDistTextClassification2x2)