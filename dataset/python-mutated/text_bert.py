import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from config import Config

class OurTokenizer(Tokenizer):

    def _tokenize(self, text):
        if False:
            return 10
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

class data_generator:

    def __init__(self, data, tokenizer, batch_size=16):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.steps

    def __iter__(self):
        if False:
            return 10
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            (X1, X2, Y) = ([], [], [])
            for i in idxs:
                d = self.data[i]
                text = d[0][:Config.bert.maxlen]
                (x1, x2) = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield ([X1, X2], Y)
                    [X1, X2, Y] = ([], [], [])

def seq_padding(X, padding=0):
    if False:
        for i in range(10):
            print('nop')
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
if __name__ == '__main__':
    tb = TextBert()
    model = tb.build_model()
    tokenizer = OurTokenizer(tb.token_dict)
    (train_data, valid_data) = tb.prepare_data()
    train_D = data_generator(train_data, tokenizer)
    valid_D = data_generator(valid_data, tokenizer)
    model.fit_generator(train_D.__iter__(), steps_per_epoch=len(train_D), epochs=5, validation_data=valid_D.__iter__(), validation_steps=len(valid_D))

class TextBert:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.path_config = Config.bert.path_config
        self.path_checkpoint = Config.bert.path_checkpoint
        self.token_dict = {}
        with codecs.open(Config.bert.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

    def prepare_data(self):
        if False:
            i = 10
            return i + 15
        neg = pd.read_excel(Config.bert.path_neg, header=None)
        pos = pd.read_excel(Config.bert.path_pos, header=None)
        data = []
        for d in neg[0]:
            data.append((d, 0))
        for d in pos[0]:
            data.append((d, 1))
        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for (i, j) in enumerate(random_order) if i % 10 != 0]
        valid_data = [data[j] for (i, j) in enumerate(random_order) if i % 10 == 0]
        return (train_data, valid_data)

    def build_model(self, m_type='bert'):
        if False:
            while True:
                i = 10
        if m_type == 'bert':
            bert_model = load_trained_model_from_checkpoint(self.path_config, self.path_checkpoint, seq_len=None)
            for l in bert_model.layers:
                l.trainable = True
            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))
            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x)
            p = Dense(1, activation='sigmoid')(x)
            model = Model([x1_in, x2_in], p)
            model.compile(loss='binary_crossentropy', optimizer=Adam(1e-05), metrics=['accuracy'])
        else:
            model = Sequential()
            model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))
            model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
            crf = CRF(len(chunk_tags), sparse_target=True)
            model.add(crf)
            model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        return model
from keras_bert import Tokenizer
token_dict = {'[CLS]': 0, '[SEP]': 1, 'un': 2, '##aff': 3, '##able': 4, '[UNK]': 5}
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))
(indices, segments) = tokenizer.encode('unaffable')
print(indices)
print(segments)
print(tokenizer.tokenize('unknown'))
(indices, segments) = tokenizer.encode('unknown')
print(tokenizer.tokenize(first='unaffable', second='钢'))
(indices, segments) = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)
print(segments)
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
sentence_pairs = [[['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']], [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']], [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']]]
token_dict = get_base_dict()
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())
model = get_model(token_num=len(token_dict), head_num=5, transformer_num=12, embed_dim=25, feed_forward_dim=100, seq_len=20, pos_num=20, dropout_rate=0.05)
compile_model(model)
model.summary()

def _generator():
    if False:
        return 10
    while True:
        yield gen_batch_inputs(sentence_pairs, token_dict, token_list, seq_len=20, mask_rate=0.3, swap_sentence_rate=1.0)
model.fit_generator(generator=_generator(), steps_per_epoch=1000, epochs=100, validation_data=_generator(), validation_steps=100, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
(inputs, output_layer) = get_model(token_num=len(token_dict), head_num=5, transformer_num=12, embed_dim=25, feed_forward_dim=100, seq_len=20, pos_num=20, dropout_rate=0.05, training=False, trainable=False, output_layer_num=4)
import os
from config import Config
config_path = Config.bert.path_config
checkpoint_path = Config.bert.path_checkpoint
vocab_path = Config.bert.dict_path
import codecs
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
from keras_bert import load_trained_model_from_checkpoint
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
from keras_bert import Tokenizer
tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
(indices, segments) = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
print(segments[:10])
import numpy as np
predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for (i, token) in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
token_dict_rev = {v: k for (k, v) in token_dict.items()}
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
indices = np.array([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens))])
segments = np.array([[0] * len(tokens) + [0] * (512 - len(tokens))])
masks = np.array([[0, 1, 1] + [0] * (512 - 3)])
predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()
print('Fill with: ', list(map(lambda x: token_dict_rev[x], predicts[0][1:3])))