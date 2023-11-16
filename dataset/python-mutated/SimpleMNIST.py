import argparse
import numpy as np
import sys
import os
import cntk as C
from cntk.train import Trainer, minibatch_size_schedule
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.device import cpu, try_set_default_device
from cntk.learners import adadelta, learning_parameter_schedule_per_sample
from cntk.ops import relu, element_times, constant
from cntk.layers import Dense, Sequential, For
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter
abs_path = os.path.dirname(os.path.abspath(__file__))

def check_path(path):
    if False:
        print('Hello World!')
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(os.path.dirname(path), '..', 'README.md'))
        raise RuntimeError("File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

def create_reader(path, is_training, input_dim, label_dim):
    if False:
        while True:
            i = 10
    return MinibatchSource(CTFDeserializer(path, StreamDefs(features=StreamDef(field='features', shape=input_dim, is_sparse=False), labels=StreamDef(field='labels', shape=label_dim, is_sparse=False))), randomize=is_training, max_sweeps=INFINITELY_REPEAT if is_training else 1)

def simple_mnist(tensorboard_logdir=None):
    if False:
        return 10
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200
    feature = C.input_variable(input_dim, np.float32)
    label = C.input_variable(num_output_classes, np.float32)
    scaled_input = element_times(constant(0.00390625), feature)
    z = Sequential([For(range(num_hidden_layers), lambda i: Dense(hidden_layers_dim, activation=relu)), Dense(num_output_classes)])(scaled_input)
    ce = cross_entropy_with_softmax(z, label)
    pe = classification_error(z, label)
    data_dir = os.path.join(abs_path, '..', '..', '..', 'DataSets', 'MNIST')
    path = os.path.normpath(os.path.join(data_dir, 'Train-28x28_cntk_text.txt'))
    check_path(path)
    reader_train = create_reader(path, True, input_dim, num_output_classes)
    input_map = {feature: reader_train.streams.features, label: reader_train.streams.labels}
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10
    progress_writers = [ProgressPrinter(tag='Training', num_epochs=num_sweeps_to_train_with)]
    if tensorboard_logdir is not None:
        progress_writers.append(TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z))
    lr = learning_parameter_schedule_per_sample(1)
    trainer = Trainer(z, (ce, pe), adadelta(z.parameters, lr), progress_writers)
    training_session(trainer=trainer, mb_source=reader_train, mb_size=minibatch_size, model_inputs_to_streams=input_map, max_samples=num_samples_per_sweep * num_sweeps_to_train_with, progress_frequency=num_samples_per_sweep).train()
    path = os.path.normpath(os.path.join(data_dir, 'Test-28x28_cntk_text.txt'))
    check_path(path)
    reader_test = create_reader(path, False, input_dim, num_output_classes)
    input_map = {feature: reader_test.streams.features, label: reader_test.streams.labels}
    C.debugging.start_profiler()
    C.debugging.enable_profiler()
    C.debugging.set_node_timing(True)
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error
    C.debugging.stop_profiler()
    trainer.print_node_timing()
    return test_result / num_minibatches_to_test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', required=False, default=None)
    args = vars(parser.parse_args())
    error = simple_mnist(args['tensorboard_logdir'])
    print('Error: %f' % error)