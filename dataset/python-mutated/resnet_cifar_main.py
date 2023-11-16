"""Runs a ResNet model on the Cifar-10 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from absl import app as absl_app
import tensorflow as tf
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.vision.image_classification import cifar_preprocessing
from official.vision.image_classification import common
from official.vision.image_classification import resnet_cifar_model
LR_SCHEDULE = [(0.1, 91), (0.01, 136), (0.001, 182)]

def learning_rate_schedule(current_epoch, current_batch, batches_per_epoch, batch_size):
    if False:
        return 10
    'Handles linear scaling rule and LR decay.\n\n  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the\n  provided scaling factor.\n\n  Args:\n    current_epoch: integer, current epoch indexed from 0.\n    current_batch: integer, current batch in the current epoch, indexed from 0.\n    batches_per_epoch: integer, number of steps in an epoch.\n    batch_size: integer, total batch sized.\n\n  Returns:\n    Adjusted learning rate.\n  '
    del current_batch, batches_per_epoch
    initial_learning_rate = common.BASE_LEARNING_RATE * batch_size / 128
    learning_rate = initial_learning_rate
    for (mult, start_epoch) in LR_SCHEDULE:
        if current_epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    return learning_rate

def run(flags_obj):
    if False:
        for i in range(10):
            print('nop')
    'Run ResNet Cifar-10 training and eval loop using native Keras APIs.\n\n  Args:\n    flags_obj: An object containing parsed flag values.\n\n  Raises:\n    ValueError: If fp16 is passed as it is not currently supported.\n\n  Returns:\n    Dictionary of training and eval stats.\n  '
    keras_utils.set_session_config(enable_eager=flags_obj.enable_eager, enable_xla=flags_obj.enable_xla)
    if flags_obj.tf_gpu_thread_mode:
        common.set_gpu_thread_mode_and_count(flags_obj)
    common.set_cudnn_batchnorm_mode()
    dtype = flags_core.get_tf_dtype(flags_obj)
    if dtype == 'fp16':
        raise ValueError('dtype fp16 is not supported in Keras. Use the default value(fp32).')
    data_format = flags_obj.data_format
    if data_format is None:
        data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
    tf.keras.backend.set_image_data_format(data_format)
    strategy = distribution_utils.get_distribution_strategy(distribution_strategy=flags_obj.distribution_strategy, num_gpus=flags_obj.num_gpus, num_workers=distribution_utils.configure_cluster(), all_reduce_alg=flags_obj.all_reduce_alg, num_packs=flags_obj.num_packs)
    if strategy:
        strategy.extended.experimental_enable_get_next_as_optional = flags_obj.enable_get_next_as_optional
    strategy_scope = distribution_utils.get_strategy_scope(strategy)
    if flags_obj.use_synthetic_data:
        distribution_utils.set_up_synthetic_data()
        input_fn = common.get_synth_input_fn(height=cifar_preprocessing.HEIGHT, width=cifar_preprocessing.WIDTH, num_channels=cifar_preprocessing.NUM_CHANNELS, num_classes=cifar_preprocessing.NUM_CLASSES, dtype=flags_core.get_tf_dtype(flags_obj), drop_remainder=True)
    else:
        distribution_utils.undo_set_up_synthetic_data()
        input_fn = cifar_preprocessing.input_fn
    train_input_dataset = input_fn(is_training=True, data_dir=flags_obj.data_dir, batch_size=flags_obj.batch_size, num_epochs=flags_obj.train_epochs, parse_record_fn=cifar_preprocessing.parse_record, datasets_num_private_threads=flags_obj.datasets_num_private_threads, dtype=dtype, drop_remainder=not flags_obj.enable_get_next_as_optional)
    eval_input_dataset = None
    if not flags_obj.skip_eval:
        eval_input_dataset = input_fn(is_training=False, data_dir=flags_obj.data_dir, batch_size=flags_obj.batch_size, num_epochs=flags_obj.train_epochs, parse_record_fn=cifar_preprocessing.parse_record)
    with strategy_scope:
        optimizer = common.get_optimizer()
        model = resnet_cifar_model.resnet56(classes=cifar_preprocessing.NUM_CLASSES)
        if flags_obj.force_v2_in_keras_compile is not None:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'] if flags_obj.report_accuracy_metrics else None, run_eagerly=flags_obj.run_eagerly, experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)
        else:
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'] if flags_obj.report_accuracy_metrics else None, run_eagerly=flags_obj.run_eagerly)
    callbacks = common.get_callbacks(learning_rate_schedule, cifar_preprocessing.NUM_IMAGES['train'])
    train_steps = cifar_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size
    train_epochs = flags_obj.train_epochs
    if flags_obj.train_steps:
        train_steps = min(flags_obj.train_steps, train_steps)
        train_epochs = 1
    num_eval_steps = cifar_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size
    validation_data = eval_input_dataset
    if flags_obj.skip_eval:
        if flags_obj.set_learning_phase_to_train:
            tf.keras.backend.set_learning_phase(1)
        num_eval_steps = None
        validation_data = None
    if not strategy and flags_obj.explicit_gpu_placement:
        no_dist_strat_device = tf.device('/device:GPU:0')
        no_dist_strat_device.__enter__()
    history = model.fit(train_input_dataset, epochs=train_epochs, steps_per_epoch=train_steps, callbacks=callbacks, validation_steps=num_eval_steps, validation_data=validation_data, validation_freq=flags_obj.epochs_between_evals, verbose=2)
    eval_output = None
    if not flags_obj.skip_eval:
        eval_output = model.evaluate(eval_input_dataset, steps=num_eval_steps, verbose=2)
    if not strategy and flags_obj.explicit_gpu_placement:
        no_dist_strat_device.__exit__()
    stats = common.build_stats(history, eval_output, callbacks)
    return stats

def define_cifar_flags():
    if False:
        i = 10
        return i + 15
    common.define_keras_flags(dynamic_loss_scale=False)
    flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin', model_dir='/tmp/cifar10_model', train_epochs=182, epochs_between_evals=10, batch_size=128)

def main(_):
    if False:
        for i in range(10):
            print('nop')
    with logger.benchmark_context(flags.FLAGS):
        return run(flags.FLAGS)
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    define_cifar_flags()
    absl_app.run(main)