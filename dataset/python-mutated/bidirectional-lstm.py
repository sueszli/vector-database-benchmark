import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, Activation, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os, argparse, itertools
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import sys
from operator import itemgetter
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.keras.backend as K

class ReportMetric(tensorflow.keras.callbacks.Callback):

    def __init__(self, valid_data, label_dim, bs, file):
        if False:
            return 10
        self.valid_data = valid_data
        self.label_dim = label_dim
        self.bs = bs
        self.file = file

    def on_epoch_end(self, batch, logs={}):
        if False:
            for i in range(10):
                print('nop')
        if self.label_dim != 1:
            Y_pred = np.argmax(model.predict(self.valid_data[0], batch_size=self.bs, verbose=1), axis=-1).reshape(-1)
            Y_true = np.argmax(self.valid_data[1], axis=2).reshape(-1)
        else:
            Y_pred = (model.predict(self.valid_data[0], batch_size=self.bs, verbose=1) > 0.5).astype('int32').reshape(-1)
            Y_true = self.valid_data[1].reshape(-1)
        report = classification_report(Y_true, Y_pred, digits=4)
        conf_matrix = confusion_matrix(Y_true, Y_pred)
        self.file.write('\n' + report + '\n')
        print(report)
        print(conf_matrix)

def make_sequences(Xs, Ys, seqlen, step=1):
    if False:
        while True:
            i = 10
    (Xseq, Yseq) = ([], [])
    for i in range(0, Xs.shape[0] - seqlen + 1, step):
        Xseq.append(Xs[i:i + seqlen])
        Yseq.append(Ys[i:i + seqlen])
    return (np.array(Xseq), np.array(Yseq))

def lstm_model(input_dim, output_dim, seq_len, hidden=128, dropout=0.0, lr=0.1):
    if False:
        i = 10
        return i + 15
    model = Sequential()
    layers = {'input': input_dim, 'hidden': hidden, 'output': output_dim}
    model.add(Bidirectional(LSTM(layers['input'], return_sequences=True), merge_mode='concat', input_shape=(seq_len, layers['input'])))
    model.add(Dropout(dropout))
    activation = 'softmax' if output_dim > 1 else 'sigmoid'
    loss = 'categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy'
    model.add(TimeDistributed(Dense(layers['output'], activation=activation)))
    model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=['acc'])
    model.summary()
    return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM on a preprocessed dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset directory')
    parser.add_argument('-i', '--iters', default=20, type=int, help='Number of random search samples')
    flags = parser.parse_args()
    dataset_dir = flags.dataset
    dataset_filenames = ['Xs_train.npy', 'Xs_val.npy', 'Xs_test.npy', 'Xs_train_val.npy', 'Xs_train_test.npy', 'Ys_train.npy', 'Ys_val.npy', 'Ys_test.npy', 'Ys_train_val.npy', 'Ys_train_test.npy']
    dataset_filenames = map(lambda x: os.path.join(dataset_dir, x), dataset_filenames)
    (X_train, X_valid, X_test, Xs_train_val, Xs_train_test, Y_train, Y_valid, Y_test, Ys_train_val, Ys_train_test) = map(np.load, dataset_filenames)
    print(X_train.shape, Y_train.shape)
    print(X_valid.shape, Y_valid.shape)
    print(X_test.shape, Y_test.shape)
    (n_label, label_dim) = Y_train.shape
    sums = np.sum(Y_train, axis=0)
    print('Label dimension is: {}'.format(label_dim))
    if not label_dim in [1, 8, 36]:
        raise Exception('Unknown label dimension! Was {}'.format(label_dim))
    input_dim = X_train.shape[1]
    print('Input dimension is: {}'.format(input_dim))
    iters = flags.iters
    print('Number of iterations is: {}'.format(iters))
    learning_rate = lambda : 10 ** np.random.uniform(-3, -1)
    sequence_length = lambda : np.random.choice([4])
    hidden_layer_size = lambda : int(2 ** np.random.uniform(6, 8))
    batch_size = lambda : int(2 ** np.random.uniform(6, 8))
    dropout = lambda : np.random.uniform(0.0, 0.2)
    step = lambda : 4
    n_epochs = lambda : 50
    ranges = [learning_rate, batch_size, n_epochs, sequence_length, dropout, hidden_layer_size, step]
    with open(os.path.join(dataset_dir, 'aggregate.txt'), 'w') as f:
        f.write('Learning Rate,Batch size,Number of epochs,Sequence length,Dropout rate,Hidden layer size\n')
    for iterations in range(iters):
        hyperparams = [p() for p in ranges]
        (lr, bs, ne, sl, dr, hls, step) = hyperparams
        hparams = ','.join(map(str, hyperparams))
        print('New hyperparameters setting:')
        print(hparams)
        output_dir = os.path.join(dataset_dir, 'results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = lstm_model(input_dim, label_dim, sl, hidden=hls, dropout=dr, lr=lr)
        print('Preparing sequences... Length = {}'.format(sl))
        (X_train_seq, Y_train_seq) = make_sequences(X_train, Y_train, sl, step)
        (X_valid_seq, Y_valid_seq) = make_sequences(X_valid, Y_valid, sl, step)
        (X_test_seq, Y_test_seq) = make_sequences(X_test, Y_test, sl, step)
        print(X_train_seq.shape, Y_train_seq.shape)
        print(X_valid_seq.shape, Y_valid_seq.shape)
        print(X_test_seq.shape, Y_test_seq.shape)
        output_file = os.path.join(output_dir, '-'.join(map(str, hyperparams)) + '.txt')
        with open(output_file, 'w') as f:
            log_dir = os.path.join(output_dir, '_'.join(map(str, hyperparams)))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            chpt_filepath = os.path.join(log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
            smcb = tensorflow.keras.callbacks.ModelCheckpoint(chpt_filepath)
            tbcb = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
            valid_data = (X_valid_seq, Y_valid_seq)
            rmcb = ReportMetric(valid_data, label_dim, bs, f)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
            h = model.fit(X_train_seq, Y_train_seq, validation_data=(X_valid_seq, Y_valid_seq), batch_size=bs, epochs=ne, callbacks=[tbcb, smcb, rmcb, reduce_lr])
            for (k, v) in h.history.items():
                f.write(k + ':' + ','.join(map(str, v)) + '\n')