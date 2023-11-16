import os
import math
import argparse
import numpy as np
import cntk as C
from InceptionV3 import inception_v3_norm_model
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, '..', '..', '..', '..', 'DataSets', 'ImageNet', 'test_data')
config_path = abs_path
model_path = os.path.join(abs_path, 'Models')
log_dir = None
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
NUM_CHANNELS = 3
NUM_CLASSES = 1000
model_name = 'InceptionV3.model'

def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if False:
        return 10
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." % map_file)
    transforms = []
    if is_training:
        transforms += [C.io.transforms.crop(crop_type='randomarea', area_ratio=(0.05, 1.0), aspect_ratio=(0.75, 1.0), jitter_type='uniratio'), C.io.transforms.scale(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=NUM_CHANNELS, interpolations='linear'), C.io.transforms.color(brightness_radius=0.125, contrast_radius=0.5, saturation_radius=0.5)]
    else:
        transforms += [C.io.transforms.crop(crop_type='center', side_ratio=0.875), C.io.transforms.scale(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=NUM_CHANNELS, interpolations='linear')]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(features=C.io.StreamDef(field='image', transforms=transforms), labels=C.io.StreamDef(field='label', shape=NUM_CLASSES))), randomize=is_training, max_samples=total_number_of_samples, multithreaded_deserializer=True)

def create_inception_v3():
    if False:
        while True:
            i = 10
    feature_var = C.ops.input_variable((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    label_var = C.ops.input_variable(NUM_CLASSES)
    drop_rate = 0.2
    bn_time_const = 4096
    out = inception_v3_norm_model(feature_var, NUM_CLASSES, drop_rate, bn_time_const)
    aux_weight = 0.3
    ce_aux = C.losses.cross_entropy_with_softmax(out['aux'], label_var)
    ce_z = C.losses.cross_entropy_with_softmax(out['z'], label_var)
    ce = C.ops.plus(C.ops.element_times(ce_aux, aux_weight), ce_z)
    pe = C.metrics.classification_error(out['z'], label_var)
    pe5 = C.metrics.classification_error(out['z'], label_var, topN=5)
    C.logging.log_number_of_parameters(out['z'])
    print()
    return {'feature': feature_var, 'label': label_var, 'ce': ce, 'pe': pe, 'pe5': pe5, 'output': out['z'], 'outputAux': out['aux']}

def create_trainer(network, epoch_size, num_epochs, minibatch_size):
    if False:
        print('Hello World!')
    initial_learning_rate = 0.45
    initial_learning_rate *= minibatch_size / 32
    learn_rate_adjust_interval = 2
    learn_rate_decrease_factor = 0.94
    lr_per_mb = []
    learning_rate = initial_learning_rate
    for i in range(0, num_epochs, learn_rate_adjust_interval):
        lr_per_mb.extend([learning_rate] * learn_rate_adjust_interval)
        learning_rate *= learn_rate_decrease_factor
    lr_schedule = C.learners.learning_parameter_schedule(lr_per_mb, epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_schedule(0.9)
    l2_reg_weight = 0.0001
    learner = C.learners.nesterov(network['ce'].parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    return C.train.Trainer(network['output'], (network['ce'], network['pe']), learner)

def train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir, testing_parameters):
    if False:
        return 10
    input_map = {network['feature']: train_source.streams.features, network['label']: train_source.streams.labels}
    if profiler_dir:
        C.debugging.start_profiler(profiler_dir, True)
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = train_source.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count
            progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)
        network['output'].save(os.path.join(model_path, 'BN-Inception_{}.model'.format(epoch)))
        C.debugging.enable_profiler()
    if profiler_dir:
        C.debugging.stop_profiler()
    (test_epoch_size, test_minibatch_size) = testing_parameters
    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0
    while sample_count < test_epoch_size:
        current_minibatch = min(test_minibatch_size, test_epoch_size - sample_count)
        data = test_source.next_minibatch(current_minibatch, input_map=input_map)
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        sample_count += data[network['label']].num_samples
        minibatch_index += 1
    print('')
    print('Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}'.format(minibatch_index + 1, metric_numer * 100.0 / metric_denom, metric_denom))
    print('')
    return metric_numer / metric_denom

def inception_v3_train_and_eval(train_data, test_data, minibatch_size=32, epoch_size=1281167, max_epochs=300, restore=True, log_to_file=None, num_mbs_per_log=100, gen_heartbeat=False, profiler_dir=None, testing_parameters=(5000, 32)):
    if False:
        i = 10
        return i + 15
    C.debugging.set_computation_network_trace_level(1)
    progress_printer = C.logging.ProgressPrinter(freq=num_mbs_per_log, tag='Training', log_to_file=log_to_file, gen_heartbeat=gen_heartbeat, num_epochs=max_epochs)
    network = create_inception_v3()
    trainer = create_trainer(network, epoch_size, max_epochs, minibatch_size)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    return train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir, testing_parameters)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the cifar-10 dataset is located', required=False, default=data_path)
    parser.add_argument('-configdir', '--configdir', help='Config directory where this python script is located', required=False, default=config_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-profilerdir', '--profilerdir', help='Directory for saving profiler output', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='300')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='32')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='1281167')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help='Force to run the script on a specified device', required=False, default=None)
    args = vars(parser.parse_args())
    if args['outputdir'] is not None:
        model_path = args['outputdir'] + '/models'
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['profilerdir'] is not None:
        profiler_dir = args['profilerdir']
    if args['device'] is not None:
        C.device.try_set_default_device(C.device.gpu(args['device']))
    data_path = args['datadir']
    if not os.path.isdir(data_path):
        raise RuntimeError('Directory %s does not exist' % data_path)
    os.chdir(data_path)
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'val_map.txt')
    inception_v3_train_and_eval(train_data, test_data, minibatch_size=args['minibatch_size'], epoch_size=args['epoch_size'], max_epochs=args['num_epochs'], restore=not args['restart'], log_to_file=args['logdir'], num_mbs_per_log=100, gen_heartbeat=True, profiler_dir=args['profilerdir'])