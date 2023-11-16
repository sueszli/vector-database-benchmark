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
        for i in range(10):
            print('nop')
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(x_train[8])
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
            for i in range(10):
                print('nop')
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn_cell0 = layers.GRUCell(units, dropout=0.5)
        self.rnn_cell1 = layers.GRUCell(units, dropout=0.5)
        self.outlayer = Sequential([layers.Dense(units), layers.Dropout(rate=0.5), layers.ReLU(), layers.Dense(1)])

    def call(self, inputs, training=None):
        if False:
            print('Hello World!')
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            (out0, state0) = self.rnn_cell0(word, state0, training)
            (out1, state1) = self.rnn_cell1(out0, state1, training)
        x = self.outlayer(out1, training)
        prob = tf.sigmoid(x)
        return prob

def main():
    if False:
        return 10
    units = 64
    epochs = 50
    model = MyRNN(units)
    model.compile(optimizer=optimizers.RMSprop(0.001), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
if __name__ == '__main__':
    main()