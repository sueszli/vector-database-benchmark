from optparse import OptionParser
import sys

def load_imdb():
    if False:
        while True:
            i = 10
    '\n    Load IMDB dataset\n    Transform input data into an RDD of Sample\n    '
    from keras.preprocessing import sequence
    from keras.datasets import imdb
    ((X_train, y_train), (X_test, y_test)) = imdb.load_data(nb_words=20000)
    X_train = sequence.pad_sequences(X_train, maxlen=100)
    X_test = sequence.pad_sequences(X_test, maxlen=100)
    return (X_train, y_train, X_test, y_test)

def build_keras_model():
    if False:
        for i in range(10):
            print('nop')
    '\n    Define a recurrent convolutional model in Keras 1.2.2\n    '
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Embedding
    from keras.layers import LSTM
    from keras.layers import Convolution1D, MaxPooling1D
    keras_model = Sequential()
    keras_model.add(Embedding(20000, 128, input_length=100))
    keras_model.add(Dropout(0.25))
    keras_model.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='valid', activation='relu', subsample_length=1))
    keras_model.add(MaxPooling1D(pool_length=4))
    keras_model.add(LSTM(70))
    keras_model.add(Dense(1))
    keras_model.add(Activation('sigmoid'))
    return keras_model
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-b', '--batchSize', type=int, dest='batchSize', default='32')
    parser.add_option('-m', '--max_epoch', type=int, dest='max_epoch', default='2')
    parser.add_option('--optimizerVersion', dest='optimizerVersion', default='optimizerV1')
    (options, args) = parser.parse_args(sys.argv)
    keras_model = build_keras_model()
    hdf5_path = '/tmp/imdb.h5'
    keras_model.save(hdf5_path)
    from bigdl.dllib.utils.common import *
    from bigdl.dllib.nn.layer import *
    from bigdl.dllib.optim.optimizer import *
    from bigdl.dllib.nn.criterion import *
    bigdl_model = Model.load_keras(hdf5_path=hdf5_path)
    sc = get_spark_context(conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    set_optimizer_version(options.optimizerVersion)
    (X_train, y_train, X_test, y_test) = load_imdb()
    train_data = to_sample_rdd(X_train, y_train)
    test_data = to_sample_rdd(X_test, y_test)
    optimizer = Optimizer(model=bigdl_model, training_rdd=train_data, criterion=BCECriterion(), optim_method=Adam(), end_trigger=MaxEpoch(options.max_epoch), batch_size=options.batchSize)
    optimizer.set_validation(batch_size=options.batchSize, val_rdd=test_data, trigger=EveryEpoch(), val_method=[Top1Accuracy()])
    optimizer.optimize()