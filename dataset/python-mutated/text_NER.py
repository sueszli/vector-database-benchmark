import pickle
import numpy as np
import pandas as pd
import platform
from collections import Counter
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
'\n# padding: pre(默认) 向前补充0  post 向后补充0\n# truncating: 文本超过 pad_num,  pre(默认) 删除前面  post 删除后面\n# x_train = pad_sequences(x, maxlen=pad_num, value=0, padding=\'post\', truncating="post")\n# print("--- ", x_train[0][:20])\n\n使用keras_bert、keras_contrib的crf时bug记录\nTypeError: Tensors in list passed to \'values\' of \'ConcatV2\' Op have types [bool, float32] that don\'t all match\n解决方案, 修改crf.py 516行：\nmask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1),\n为:\nmask2 = K.cast(K.concatenate([mask, K.cast(K.zeros_like(mask[:, :1]), mask.dtype)], axis=1),\n'
from keras.preprocessing.sequence import pad_sequences
from config.setting import Config

def load_data():
    if False:
        i = 10
        return i + 15
    train = _parse_data(Config.nlp_ner.path_train)
    test = _parse_data(Config.nlp_ner.path_test)
    print('--- init 数据加载解析完成 ---')
    word_counts = Counter((row[0].lower() for sample in train for row in sample))
    vocab = [w for (w, f) in iter(word_counts.items()) if f >= 2]
    chunk_tags = Config.nlp_ner.chunk_tags
    with open(Config.nlp_ner.path_config, 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)
    print('--- init 配置文件保存成功 ---')
    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    print('--- init 对数据进行编码，生成训练需要的数据格式 ---')
    return (train, test, (vocab, chunk_tags))

def _parse_data(filename):
    if False:
        i = 10
        return i + 15
    "\n    以单下划线开头（_foo）的代表不能直接访问的类属性\n    用于解析数据，用于模型训练\n    :param filename: 文件地址\n    :return: data: 解析数据后的结果\n    [[['中', 'B-ORG'], ['共', 'I-ORG']], [['中', 'B-ORG'], ['国', 'I-ORG']]]\n    "
    with open(filename, 'rb') as fn:
        split_text = '\n'
        texts = fn.read().decode('utf-8').strip().split(split_text + split_text)
        data = [[[' ', 'O'] if len(row.split()) != 2 else row.split() for row in text.split(split_text) if len(row) > 0] for text in texts]
    return data

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if False:
        i = 10
        return i + 15
    if maxlen is None:
        maxlen = max((len(s) for s in data))
    word2idx = dict(((w, i) for (i, w) in enumerate(vocab)))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen)
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return (x, y_chunk)

def process_data(data, vocab, maxlen=100):
    if False:
        return 10
    word2idx = dict(((w, i) for (i, w) in enumerate(vocab)))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)
    return (x, length)

def create_model(len_vocab, len_chunk_tags):
    if False:
        for i in range(10):
            print('nop')
    model = Sequential()
    model.add(Embedding(len_vocab, Config.nlp_ner.EMBED_DIM, mask_zero=True))
    model.add(Bidirectional(LSTM(Config.nlp_ner.BiLSTM_UNITS // 2, return_sequences=True)))
    model.add(Dropout(0.25))
    crf = CRF(len_chunk_tags, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    return model

def train():
    if False:
        return 10
    ((train_x, train_y), (test_x, test_y), (vocab, chunk_tags)) = load_data()
    model = create_model(len(vocab), len(chunk_tags))
    model.fit(train_x, train_y, batch_size=16, epochs=Config.nlp_ner.EPOCHS, validation_data=[test_x, test_y])
    model.save(Config.nlp_ner.path_model)

def test():
    if False:
        return 10
    with open(Config.nlp_ner.path_config, 'rb') as inp:
        (vocab, chunk_tags) = pickle.load(inp)
    model = create_model(len(vocab), len(chunk_tags))
    with open(Config.nlp_ner.path_origin, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for predict_text in lines:
            content = predict_text.strip()
            (text_EMBED, length) = process_data(content, vocab)
            model.load_weights(Config.nlp_ner.path_model)
            raw = model.predict(text_EMBED)[0][-length:]
            pre_result = [np.argmax(row) for row in raw]
            result_tags = [chunk_tags[i] for i in pre_result]
            result = {}
            tag_list = [i for i in chunk_tags if i not in ['O']]
            for (word, t) in zip(content, result_tags):
                if t not in tag_list:
                    continue
                for i in range(0, len(tag_list), 2):
                    if t in tag_list[i:i + 2]:
                        tag = tag_list[i].split('-')[-1]
                        if tag not in result:
                            result[tag] = ''
                        result[tag] += ' ' + word if t == tag_list[i] else word
            print(result)

def main():
    if False:
        return 10
    train()
    test()