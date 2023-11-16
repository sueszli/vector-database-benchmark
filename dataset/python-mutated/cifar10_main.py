"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import division
from __future__ import print_function
import argparse
import functools
import itertools
import os
import cifar10
import cifar10_model
import cifar10_utils
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def get_model_fn(num_gpus, variable_strategy, num_workers):
    if False:
        return 10
    'Returns a function that will build the resnet model.'

    def _resnet_model_fn(features, labels, mode, params):
        if False:
            return 10
        'Resnet model body.\n\n    Support single host, one or more GPU training. Parameter distribution can\n    be either one of the following scheme.\n    1. CPU is the parameter server and manages gradient updates.\n    2. Parameters are distributed evenly across all GPUs, and the first GPU\n       manages gradient updates.\n\n    Args:\n      features: a list of tensors, one for each tower\n      labels: a list of tensors, one for each tower\n      mode: ModeKeys.TRAIN or EVAL\n      params: Hyperparameters suitable for tuning\n    Returns:\n      A EstimatorSpec object.\n    '
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        weight_decay = params.weight_decay
        momentum = params.momentum
        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'
        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'
        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = cifar10_utils.local_device_setter(worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = cifar10_utils.local_device_setter(ps_device_type='gpu', worker_device=worker_device, ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('resnet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        (loss, gradvars, preds) = _tower_fn(is_training, weight_decay, tower_features[i], tower_labels[i], data_format, params.num_layers, params.batch_norm_decay, params.batch_norm_epsilon)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for (grad, var) in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for (var, grads) in six.iteritems(all_grads):
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1.0 / len(grads))
                gradvars.append((avg_grad, var))
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch('train') // (params.train_batch_size * num_workers)
            boundaries = [num_batches_per_epoch * x for x in np.array([82, 123, 300], dtype=np.int64)]
            staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]
            learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), boundaries, staged_lr)
            loss = tf.reduce_mean(tower_losses, name='loss')
            examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(params.train_batch_size, every_n_steps=10)
            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
            logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
            train_hooks = [logging_hook, examples_sec_hook]
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                train_hooks.append(sync_replicas_hook)
            train_op = [optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)
            predictions = {'classes': tf.concat([p['classes'] for p in tower_preds], axis=0), 'probabilities': tf.concat([p['probabilities'] for p in tower_preds], axis=0)}
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, training_hooks=train_hooks, eval_metric_ops=metrics)
    return _resnet_model_fn

def _tower_fn(is_training, weight_decay, feature, label, data_format, num_layers, batch_norm_decay, batch_norm_epsilon):
    if False:
        i = 10
        return i + 15
    'Build computation tower (Resnet).\n\n  Args:\n    is_training: true if is training graph.\n    weight_decay: weight regularization strength, a float.\n    feature: a Tensor.\n    label: a Tensor.\n    data_format: channels_last (NHWC) or channels_first (NCHW).\n    num_layers: number of layers, an int.\n    batch_norm_decay: decay for batch normalization, a float.\n    batch_norm_epsilon: epsilon for batch normalization, a float.\n\n  Returns:\n    A tuple with the loss for the tower, the gradients and parameters, and\n    predictions.\n\n  '
    model = cifar10_model.ResNetCifar10(num_layers, batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon, is_training=is_training, data_format=data_format)
    logits = model.forward_pass(feature, input_data_format='channels_last')
    tower_pred = {'classes': tf.argmax(input=logits, axis=1), 'probabilities': tf.nn.softmax(logits)}
    tower_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label)
    tower_loss = tf.reduce_mean(tower_loss)
    model_params = tf.trainable_variables()
    tower_loss += weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])
    tower_grad = tf.gradients(tower_loss, model_params)
    return (tower_loss, zip(tower_grad, model_params), tower_pred)

def input_fn(data_dir, subset, num_shards, batch_size, use_distortion_for_training=True):
    if False:
        return 10
    "Create input graph for model.\n\n  Args:\n    data_dir: Directory where TFRecords representing the dataset are located.\n    subset: one of 'train', 'validate' and 'eval'.\n    num_shards: num of towers participating in data-parallel training.\n    batch_size: total batch size for training to be divided by the number of\n    shards.\n    use_distortion_for_training: True to use distortions.\n  Returns:\n    two lists of tensors for features and labels, each of num_shards length.\n  "
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
        (image_batch, label_batch) = dataset.make_batch(batch_size)
        if num_shards <= 1:
            return ([image_batch], [label_batch])
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return (feature_shards, label_shards)

def get_experiment_fn(data_dir, num_gpus, variable_strategy, use_distortion_for_training=True):
    if False:
        for i in range(10):
            print('nop')
    'Returns an Experiment function.\n\n  Experiments perform training on several workers in parallel,\n  in other words experiments know how to invoke train and eval in a sensible\n  fashion for distributed training. Arguments passed directly to this\n  function are not tunable, all other arguments should be passed within\n  tf.HParams, passed to the enclosed function.\n\n  Args:\n      data_dir: str. Location of the data for input_fns.\n      num_gpus: int. Number of GPUs on each worker.\n      variable_strategy: String. CPU to use CPU as the parameter server\n      and GPU to use the GPUs as the parameter server.\n      use_distortion_for_training: bool. See cifar10.Cifar10DataSet.\n  Returns:\n      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->\n      tf.contrib.learn.Experiment.\n\n      Suitable for use by tf.contrib.learn.learn_runner, which will run various\n      methods on Experiment (train, evaluate) based on information\n      about the current runner in `run_config`.\n  '

    def _experiment_fn(run_config, hparams):
        if False:
            print('Hello World!')
        'Returns an Experiment.'
        train_input_fn = functools.partial(input_fn, data_dir, subset='train', num_shards=num_gpus, batch_size=hparams.train_batch_size, use_distortion_for_training=use_distortion_for_training)
        eval_input_fn = functools.partial(input_fn, data_dir, subset='eval', batch_size=hparams.eval_batch_size, num_shards=num_gpus)
        num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
        if num_eval_examples % hparams.eval_batch_size != 0:
            raise ValueError('validation set size must be multiple of eval_batch_size')
        train_steps = hparams.train_steps
        eval_steps = num_eval_examples // hparams.eval_batch_size
        classifier = tf.estimator.Estimator(model_fn=get_model_fn(num_gpus, variable_strategy, run_config.num_worker_replicas or 1), config=run_config, params=hparams)
        return tf.contrib.learn.Experiment(classifier, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, train_steps=train_steps, eval_steps=eval_steps)
    return _experiment_fn

def main(job_dir, data_dir, num_gpus, variable_strategy, use_distortion_for_training, log_device_placement, num_intra_threads, **hparams):
    if False:
        print('Hello World!')
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement, intra_op_parallelism_threads=num_intra_threads, gpu_options=tf.GPUOptions(force_gpu_compatible=True))
    config = cifar10_utils.RunConfig(session_config=sess_config, model_dir=job_dir)
    tf.contrib.learn.learn_runner.run(get_experiment_fn(data_dir, num_gpus, variable_strategy, use_distortion_for_training), run_config=config, hparams=tf.contrib.training.HParams(is_chief=config.is_chief, **hparams))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument('--job-dir', type=str, required=True, help='The directory where the model will be stored.')
    parser.add_argument('--variable-strategy', choices=['CPU', 'GPU'], type=str, default='CPU', help='Where to locate variable operations')
    parser.add_argument('--num-gpus', type=int, default=1, help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument('--num-layers', type=int, default=44, help='The number of layers of the model.')
    parser.add_argument('--train-steps', type=int, default=80000, help='The number of steps to use for training.')
    parser.add_argument('--train-batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--eval-batch-size', type=int, default=100, help='Batch size for validation.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.0002, help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='      This is the inital learning rate value. The learning rate will decrease\n      during training. For more details check the model_fn implementation in\n      this file.      ')
    parser.add_argument('--use-distortion-for-training', type=bool, default=True, help='If doing image distortion for training.')
    parser.add_argument('--sync', action='store_true', default=False, help='      If present when running in a distributed environment will run on sync mode.      ')
    parser.add_argument('--num-intra-threads', type=int, default=0, help='      Number of threads to use for intra-op parallelism. When training on CPU\n      set to 0 to have the system pick the appropriate number or alternatively\n      set it to the number of physical CPU cores.      ')
    parser.add_argument('--num-inter-threads', type=int, default=0, help='      Number of threads to use for inter-op parallelism. If set to 0, the\n      system will pick an appropriate number.      ')
    parser.add_argument('--data-format', type=str, default=None, help='      If not set, the data format best for the training device is used. \n      Allowed values: channels_first (NCHW) channels_last (NHWC).      ')
    parser.add_argument('--log-device-placement', action='store_true', default=False, help='Whether to log device placement.')
    parser.add_argument('--batch-norm-decay', type=float, default=0.997, help='Decay for batch norm.')
    parser.add_argument('--batch-norm-epsilon', type=float, default=1e-05, help='Epsilon for batch norm.')
    args = parser.parse_args()
    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), 'Requested GPUs but none found.'
    if args.num_gpus < 0:
        raise ValueError('Invalid GPU count: "--num-gpus" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set--variable-strategy=CPU.')
    if (args.num_layers - 2) % 6 != 0:
        raise ValueError('Invalid --num-layers parameter.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')
    main(**vars(args))