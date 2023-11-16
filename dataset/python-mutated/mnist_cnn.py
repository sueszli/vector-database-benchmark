from optparse import OptionParser
import sys
from bigdl.dllib.nn.keras.keras_utils import *
import keras.backend
if keras.backend.image_dim_ordering() == 'th':
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)

def get_mnist(sc, data_type='train', location='/tmp/mnist'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Download or load MNIST dataset to/from the specified path.\n    Normalize and transform input data into an RDD of Sample\n    '
    from bigdl.dllib.feature.dataset import mnist
    from bigdl.dllib.feature.dataset.transformer import normalizer
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = images.reshape((images.shape[0],) + input_shape)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1)
    record = images.zip(labels).map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD), rec_tuple[1])).map(lambda t: Sample.from_ndarray(t[0], t[1]))
    return record

def build_keras_model():
    if False:
        for i in range(10):
            print('nop')
    '\n    Define a convnet model in Keras 1.2.2\n    '
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    keras_model = Sequential()
    keras_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
    keras_model.add(Activation('relu'))
    keras_model.add(Convolution2D(32, 3, 3))
    keras_model.add(Activation('relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Dropout(0.25))
    keras_model.add(Flatten())
    keras_model.add(Dense(128))
    keras_model.add(Activation('relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(10))
    keras_model.add(Activation('softmax'))
    return keras_model
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-b', '--batchSize', type=int, dest='batchSize', default='128')
    parser.add_option('-m', '--max_epoch', type=int, dest='max_epoch', default='12')
    parser.add_option('-d', '--dataPath', dest='dataPath', default='/tmp/mnist')
    parser.add_option('--optimizerVersion', dest='optimizerVersion', default='optimizerV1')
    (options, args) = parser.parse_args(sys.argv)
    keras_model = build_keras_model()
    json_path = '/tmp/lenet.json'
    save_keras_definition(keras_model, json_path)
    from bigdl.dllib.utils.common import *
    from bigdl.dllib.nn.layer import *
    from bigdl.dllib.optim.optimizer import *
    from bigdl.dllib.nn.criterion import *
    bigdl_model = Model.load_keras(json_path=json_path)
    sc = get_spark_context(conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()
    set_optimizer_version(options.optimizerVersion)
    train_data = get_mnist(sc, 'train', options.dataPath)
    test_data = get_mnist(sc, 'test', options.dataPath)
    optimizer = Optimizer(model=bigdl_model, training_rdd=train_data, criterion=ClassNLLCriterion(logProbAsInput=False), optim_method=Adadelta(), end_trigger=MaxEpoch(options.max_epoch), batch_size=options.batchSize)
    optimizer.set_validation(batch_size=options.batchSize, val_rdd=test_data, trigger=EveryEpoch(), val_method=[Top1Accuracy()])
    optimizer.optimize()