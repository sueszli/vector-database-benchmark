import re
import os
import keras
import random
import gensim
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Flatten, Bidirectional, Embedding, GRU, Input, multiply
'\n# padding: pre(默认) 向前补充0  post 向后补充0\n# truncating: 文本超过 pad_num,  pre(默认) 删除前面  post 删除后面\n# x_train = pad_sequences(x, maxlen=pad_num, value=0, padding=\'post\', truncating="post")\n# print("--- ", x_train[0][:20])\n'
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from config import Config
import pickle
import matplotlib.pyplot as plt

def load_pkl(filename):
    if False:
        while True:
            i = 10
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model

def save_pkl(model, filename):
    if False:
        while True:
            i = 10
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)

def trainWord2Vec(infile, outfile):
    if False:
        for i in range(10):
            print('nop')
    sentences = gensim.models.word2vec.LineSentence(infile)
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(outfile)

def loadMyWord2Vec(outfile):
    if False:
        i = 10
        return i + 15
    Word2VecModel = gensim.models.Word2Vec.load(outfile)
    return Word2VecModel

def load_embeding():
    if False:
        return 10
    infile = './CarCommentAll_cut.csv'
    outfile = '/opt/data/nlp/开源词向量/gensim_word2vec_60/Word60.model'
    Word2VecModel = loadMyWord2Vec(outfile)
    print('空间的词向量（60 维）:', Word2VecModel.wv['空间'].shape, Word2VecModel.wv['空间'])
    print('打印与空间最相近的5个词语: ', Word2VecModel.wv.most_similar('空间', topn=5))
    vocab_list = [word for (word, Vocab) in Word2VecModel.wv.vocab.items()]
    word_index = {' ': 0}
    word_vector = {}
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = Word2VecModel.wv[word]
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]
    print('加载词向量结束..')
    return (vocab_list, word_index, embeddings_matrix)

def plot_history(history):
    if False:
        return 10
    history_dict = history.history
    print(history_dict.keys())
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Emotion_loss.png')
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Emotion_acc.png')

class EmotionModel(object):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.model = None
        self.config = config
        self.pre_num = self.config.pre_num
        self.data_file = self.config.data_file
        self.vocab_list = self.config.vocab_list
        self.word_index = self.config.word_index
        self.EMBEDDING_DIM = self.config.EMBEDDING_DIM
        self.MAX_SEQUENCE_LENGTH = self.config.MAX_SEQUENCE_LENGTH
        if os.path.exists(self.config.model_file):
            self.model = load_model(self.config.model_file)
            self.model.summary()
        else:
            self.train()

    def build_model(self, embeddings_matrix):
        if False:
            return 10
        embedding_layer = Embedding(input_dim=len(embeddings_matrix), output_dim=self.EMBEDDING_DIM, weights=[embeddings_matrix], input_length=self.MAX_SEQUENCE_LENGTH, trainable=False)
        print('开始训练模型.....')
        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        attention_probs = Dense(self.EMBEDDING_DIM, activation='softmax', name='attention_probs')(embedded_sequences)
        attention_mul = multiply([embedded_sequences, attention_probs], name='attention_mul')
        x = Bidirectional(GRU(self.EMBEDDING_DIM, return_sequences=True, dropout=0.5))(attention_mul)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        preds = Dense(self.pre_num, activation='softmax')(x)
        self.model = Model(sequence_input, preds)
        optimizer = Adam(lr=self.config.learning_rate, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_word2jieba(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_list = load_pkl(self.vocab_list)
        if vocab_list != []:
            print('加载词的总量: ', len(vocab_list))
            for word in vocab_list:
                jieba.add_word(word)

    def predict(self, line):
        if False:
            return 10
        '预测'
        word_index = load_pkl(self.word_index)
        STOPWORDS = ['-', '\t', '\n', '.', '。', ',', '，', ';', '!', '！', '?', '？', '%']
        words = [word for word in jieba.cut(str(line), cut_all=False) if word not in STOPWORDS]
        indexs = [word_index.get(word, 0) for word in words]
        x_pred = pad_sequences([indexs], maxlen=self.MAX_SEQUENCE_LENGTH)
        res = self.model.predict(x_pred, verbose=0)[0]
        return res

    def load_data(self, word_index, vocab_list, test_size=0.25):
        if False:
            for i in range(10):
                print('nop')
        STOPWORDS = ['-', '\t', '\n', '.', '。', ',', '，', ';', '!', '！', '?', '？', '%']
        if vocab_list != []:
            for word in vocab_list:
                jieba.add_word(word)

        def func(line):
            if False:
                while True:
                    i = 10
            words = [word for word in jieba.cut(str(line), cut_all=False) if word not in STOPWORDS]
            indexs = [word_index.get(word, 0) for word in words]
            return indexs
        df = pd.read_excel(self.data_file, header=0, error_bad_lines=False, encoding='utf_8_sig')
        x = df['comment'].apply(lambda line: func(line)).tolist()
        x = pad_sequences(x, maxlen=self.MAX_SEQUENCE_LENGTH)
        y = df['label'].tolist()
        '\n        In [7]: to_categorical(np.asarray([1,1,0,1,3]))\n        Out[7]:\n        array([[0., 1., 0., 0.],\n            [0., 1., 0., 0.],\n            [1., 0., 0., 0.],\n            [0., 1., 0., 0.],\n            [0., 0., 0., 1.]], dtype=float32)\n        '
        y = to_categorical(np.asarray(y))
        (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=test_size, random_state=10000)
        return ((x_train, y_train), (x_test, y_test))

    def train(self):
        if False:
            i = 10
            return i + 15
        '训练模型'
        (vocab_list, word_index, embeddings_matrix) = load_embeding()
        save_pkl(vocab_list, self.vocab_list)
        save_pkl(word_index, self.word_index)
        ((x_train, y_train), (x_test, y_test)) = self.load_data(word_index, vocab_list)
        print('---------')
        print(x_train[:3], '\n', y_train[:3])
        print('\n')
        print(x_test[:3], '\n', y_test[:3])
        print('---------')
        self.build_model(embeddings_matrix)
        history = self.model.fit(x_train, y_train, batch_size=60, epochs=40, validation_split=0.2, verbose=0)
        plot_history(history)
        self.model.evaluate(x_test, y_test, verbose=2)
        self.model.save(self.config.model_file)
if __name__ == '__main__':
    model = EmotionModel(Config)
    status = False
    while 1:
        text = input('text:')
        if text in ['exit', 'quit']:
            break
        if not status:
            model.load_word2jieba()
            status = True
        res = model.predict(text)
        label_dic = {0: '消极的', 1: '中性的', 2: '积极的'}
        print(res, ' : ', label_dic[np.argmax(res)])