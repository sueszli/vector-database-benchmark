import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Sequential
tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
batchsz = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
((x_train, y_train), (x_test, y_test)) = keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)
x_train[0]
word_index = keras.datasets.imdb.get_word_index()
word_index = {k: v + 3 for (k, v) in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    if False:
        return 10
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(x_train[8])
print('Indexing word vectors.')
embeddings_index = {}
GLOVE_DIR = 'C:\\Users\\z390\\Downloads\\glove6b50dtxt'
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))
len(embeddings_index.keys())
len(word_index.keys())
MAX_NUM_WORDS = total_words
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, embedding_len))
applied_vec_count = 0
for (word, i) in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        applied_vec_count += 1
print(applied_vec_count, embedding_matrix.shape)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

class MyRNN(keras.Model):

    def __init__(self, units):
        if False:
            i = 10
            return i + 15
        super(MyRNN, self).__init__()
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len, trainable=False)
        self.embedding.build(input_shape=(None, max_review_len))
        self.rnn = keras.Sequential([layers.LSTM(units, dropout=0.5, return_sequences=True), layers.LSTM(units, dropout=0.5)])
        self.outlayer = Sequential([layers.Dense(32), layers.Dropout(rate=0.5), layers.ReLU(), layers.Dense(1)])

    def call(self, inputs, training=None):
        if False:
            i = 10
            return i + 15
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x, training)
        prob = tf.sigmoid(x)
        return prob

def main():
    if False:
        i = 10
        return i + 15
    units = 512
    epochs = 50
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
if __name__ == '__main__':
    main()