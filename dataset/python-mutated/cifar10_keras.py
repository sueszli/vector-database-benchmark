import argparse
import logging
import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import cifar10
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop
from keras.utils import multi_gpu_model, to_categorical
import keras.backend.tensorflow_backend as KTF
import nni
from nni.algorithms.hpo.networkmorphism_tuner.graph import json_to_graph
log_format = '%(asctime)s %(message)s'
logging.basicConfig(filename='networkmorphism.log', filemode='a', level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger('cifar10-network-morphism-keras')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

def get_args():
    if False:
        while True:
            i = 10
    ' get args from command line\n    '
    parser = argparse.ArgumentParser('cifar10')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='epoch limit')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay of the learning rate')
    return parser.parse_args()
trainloader = None
testloader = None
net = None
args = get_args()
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

def build_graph_from_json(ir_model_json):
    if False:
        i = 10
        return i + 15
    'build model from json representation\n    '
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_keras_model()
    return model

def parse_rev_args(receive_msg):
    if False:
        return 10
    ' parse reveive msgs to global variable\n    '
    global trainloader
    global testloader
    global net
    logger.debug('Preparing data..')
    ((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    trainloader = (x_train, y_train)
    testloader = (x_test, y_test)
    logger.debug('Building model..')
    net = build_graph_from_json(receive_msg)
    try:
        available_devices = os.environ['CUDA_VISIBLE_DEVICES']
        gpus = len(available_devices.split(','))
        if gpus > 1:
            net = multi_gpu_model(net, gpus)
    except KeyError:
        logger.debug('parallel model not support in this config settings')
    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.learning_rate, momentum=0.9, decay=args.weight_decay)
    if args.optimizer == 'Adadelta':
        optimizer = Adadelta(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == 'Adagrad':
        optimizer = Adagrad(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == 'Adam':
        optimizer = Adam(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == 'Adamax':
        optimizer = Adamax(lr=args.learning_rate, decay=args.weight_decay)
    if args.optimizer == 'RMSprop':
        optimizer = RMSprop(lr=args.learning_rate, decay=args.weight_decay)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return 0

class SendMetrics(keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs=None):
        if False:
            print('Hello World!')
        '\n        Run on end of each epoch\n        '
        if logs is None:
            logs = dict()
        logger.debug(logs)
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

def train_eval():
    if False:
        for i in range(10):
            print('nop')
    ' train and eval the model\n    '
    global trainloader
    global testloader
    global net
    (x_train, y_train) = trainloader
    (x_test, y_test) = testloader
    net.fit(x=x_train, y=y_train, batch_size=args.batch_size, validation_data=(x_test, y_test), epochs=args.epochs, shuffle=True, callbacks=[SendMetrics(), EarlyStopping(min_delta=0.001, patience=10), TensorBoard(log_dir=TENSORBOARD_DIR)])
    (_, acc) = net.evaluate(x_test, y_test)
    logger.debug('Final result is: %.3f', acc)
    nni.report_final_result(acc)
if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)
        parse_rev_args(RCV_CONFIG)
        train_eval()
    except Exception as exception:
        logger.exception(exception)
        raise