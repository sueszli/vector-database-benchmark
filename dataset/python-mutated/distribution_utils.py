"""Helper functions for running models in a distributed setting."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random
import string
import tensorflow as tf
from official.utils.misc import tpu_lib

def _collective_communication(all_reduce_alg):
    if False:
        print('Hello World!')
    "Return a CollectiveCommunication based on all_reduce_alg.\n\n  Args:\n    all_reduce_alg: a string specifying which collective communication to pick,\n      or None.\n\n  Returns:\n    tf.distribute.experimental.CollectiveCommunication object\n\n  Raises:\n    ValueError: if `all_reduce_alg` not in [None, 'ring', 'nccl']\n  "
    collective_communication_options = {None: tf.distribute.experimental.CollectiveCommunication.AUTO, 'ring': tf.distribute.experimental.CollectiveCommunication.RING, 'nccl': tf.distribute.experimental.CollectiveCommunication.NCCL}
    if all_reduce_alg not in collective_communication_options:
        raise ValueError("When used with `multi_worker_mirrored`, valid values for all_reduce_alg are ['ring', 'nccl'].  Supplied value: {}".format(all_reduce_alg))
    return collective_communication_options[all_reduce_alg]

def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
    if False:
        for i in range(10):
            print('nop')
    "Return a CrossDeviceOps based on all_reduce_alg and num_packs.\n\n  Args:\n    all_reduce_alg: a string specifying which cross device op to pick, or None.\n    num_packs: an integer specifying number of packs for the cross device op.\n\n  Returns:\n    tf.distribute.CrossDeviceOps object or None.\n\n  Raises:\n    ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].\n  "
    if all_reduce_alg is None:
        return None
    mirrored_all_reduce_options = {'nccl': tf.distribute.NcclAllReduce, 'hierarchical_copy': tf.distribute.HierarchicalCopyAllReduce}
    if all_reduce_alg not in mirrored_all_reduce_options:
        raise ValueError("When used with `mirrored`, valid values for all_reduce_alg are ['nccl', 'hierarchical_copy'].  Supplied value: {}".format(all_reduce_alg))
    cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
    return cross_device_ops_class(num_packs=num_packs)

def get_distribution_strategy(distribution_strategy='default', num_gpus=0, num_workers=1, all_reduce_alg=None, num_packs=1, tpu_address=None):
    if False:
        print('Hello World!')
    'Return a DistributionStrategy for running the model.\n\n  Args:\n    distribution_strategy: a string specifying which distribution strategy to\n      use. Accepted values are \'off\', \'default\', \'one_device\', \'mirrored\',\n      \'parameter_server\', \'multi_worker_mirrored\', and \'tpu\' -- case insensitive.\n      \'off\' means not to use Distribution Strategy; \'default\' means to choose from\n      `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`\n      according to the number of GPUs and number of workers. \'tpu\' means to use\n      TPUStrategy using `tpu_address`.\n    num_gpus: Number of GPUs to run this model.\n    num_workers: Number of workers to run this model.\n    all_reduce_alg: Optional. Specifies which algorithm to use when performing\n      all-reduce. For `MirroredStrategy`, valid values are "nccl" and\n      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are\n      "ring" and "nccl".  If None, DistributionStrategy will choose based on\n      device topology.\n    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`\n      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.\n    tpu_address: Optional. String that represents TPU to connect to. Must not\n      be None if `distribution_strategy` is set to `tpu`.\n  Returns:\n    tf.distribute.DistibutionStrategy object.\n  Raises:\n    ValueError: if `distribution_strategy` is \'off\' or \'one_device\' and\n      `num_gpus` is larger than 1; or `num_gpus` is negative or if\n      `distribution_strategy` is `tpu` but `tpu_address` is not specified.\n  '
    if num_gpus < 0:
        raise ValueError('`num_gpus` can not be negative.')
    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == 'off':
        if num_gpus > 1:
            raise ValueError("When {} GPUs and  {} workers are specified, distribution_strategy flag cannot be set to 'off'.".format(num_gpus, num_workers))
        return None
    if distribution_strategy == 'tpu':
        cluster_resolver = tpu_lib.tpu_initialize(tpu_address)
        return tf.distribute.experimental.TPUStrategy(cluster_resolver)
    if distribution_strategy == 'multi_worker_mirrored':
        return tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=_collective_communication(all_reduce_alg))
    if distribution_strategy == 'one_device' or (distribution_strategy == 'default' and num_gpus <= 1):
        if num_gpus == 0:
            return tf.distribute.OneDeviceStrategy('device:CPU:0')
        else:
            if num_gpus > 1:
                raise ValueError('`OneDeviceStrategy` can not be used for more than one device.')
            return tf.distribute.OneDeviceStrategy('device:GPU:0')
    if distribution_strategy in ('mirrored', 'default'):
        if num_gpus == 0:
            assert distribution_strategy == 'mirrored'
            devices = ['device:CPU:0']
        else:
            devices = ['device:GPU:%d' % i for i in range(num_gpus)]
        return tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))
    if distribution_strategy == 'parameter_server':
        return tf.distribute.experimental.ParameterServerStrategy()
    raise ValueError('Unrecognized Distribution Strategy: %r' % distribution_strategy)

def per_replica_batch_size(batch_size, num_gpus):
    if False:
        return 10
    'For multi-gpu, batch-size must be a multiple of the number of GPUs.\n\n\n  Note that distribution strategy handles this automatically when used with\n  Keras. For using with Estimator, we need to get per GPU batch.\n\n  Args:\n    batch_size: Global batch size to be divided among devices. This should be\n      equal to num_gpus times the single-GPU batch_size for multi-gpu training.\n    num_gpus: How many GPUs are used with DistributionStrategies.\n\n  Returns:\n    Batch size per device.\n\n  Raises:\n    ValueError: if batch_size is not divisible by number of devices\n  '
    if num_gpus <= 1:
        return batch_size
    remainder = batch_size % num_gpus
    if remainder:
        err = 'When running with multiple GPUs, batch size must be a multiple of the number of available GPUs. Found {} GPUs with a batch size of {}; try --batch_size={} instead.'.format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)

class SyntheticDataset(object):
    """A dataset that generates synthetic data on each device."""

    def __init__(self, dataset, split_by=1):
        if False:
            print('Hello World!')
        with tf.device('device:CPU:0'):
            tensor = tf.data.experimental.get_single_element(dataset.take(1))
        flat_tensor = tf.nest.flatten(tensor)
        variable_data = []
        initializers = []
        for t in flat_tensor:
            rebatched_t = tf.split(t, num_or_size_splits=split_by, axis=0)[0]
            assert rebatched_t.shape.is_fully_defined(), rebatched_t.shape
            v = tf.compat.v1.get_local_variable(self._random_name(), initializer=rebatched_t)
            variable_data.append(v)
            initializers.append(v.initializer)
        input_data = tf.nest.pack_sequence_as(tensor, variable_data)
        self._iterator = SyntheticIterator(input_data, initializers)

    def _random_name(self, size=10, chars=string.ascii_uppercase + string.digits):
        if False:
            i = 10
            return i + 15
        return ''.join((random.choice(chars) for _ in range(size)))

    def __iter__(self):
        if False:
            print('Hello World!')
        return self._iterator

    def make_one_shot_iterator(self):
        if False:
            i = 10
            return i + 15
        return self._iterator

    def make_initializable_iterator(self):
        if False:
            i = 10
            return i + 15
        return self._iterator

class SyntheticIterator(object):
    """A dataset that generates synthetic data on each device."""

    def __init__(self, input_data, initializers):
        if False:
            return 10
        self._input_data = input_data
        self._initializers = initializers

    def get_next(self):
        if False:
            i = 10
            return i + 15
        return self._input_data

    def next(self):
        if False:
            i = 10
            return i + 15
        return self.__next__()

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.get_next()
        except tf.errors.OutOfRangeError:
            raise StopIteration

    def initialize(self):
        if False:
            print('Hello World!')
        if tf.executing_eagerly():
            return tf.no_op()
        else:
            return self._initializers

def _monkey_patch_dataset_method(strategy):
    if False:
        return 10
    "Monkey-patch `strategy`'s `make_dataset_iterator` method."

    def make_dataset(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        tf.compat.v1.logging.info('Using pure synthetic data.')
        with self.scope():
            if self.extended._global_batch_size:
                return SyntheticDataset(dataset, self.num_replicas_in_sync)
            else:
                return SyntheticDataset(dataset)

    def make_iterator(self, dataset):
        if False:
            print('Hello World!')
        dist_dataset = make_dataset(self, dataset)
        return iter(dist_dataset)
    strategy.orig_make_dataset_iterator = strategy.make_dataset_iterator
    strategy.make_dataset_iterator = make_iterator
    strategy.orig_distribute_dataset = strategy.experimental_distribute_dataset
    strategy.experimental_distribute_dataset = make_dataset

def _undo_monkey_patch_dataset_method(strategy):
    if False:
        print('Hello World!')
    if hasattr(strategy, 'orig_make_dataset_iterator'):
        strategy.make_dataset_iterator = strategy.orig_make_dataset_iterator
    if hasattr(strategy, 'orig_distribute_dataset'):
        strategy.make_dataset_iterator = strategy.orig_distribute_dataset

def set_up_synthetic_data():
    if False:
        return 10
    _monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
    _monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
    _monkey_patch_dataset_method(tf.distribute.experimental.MultiWorkerMirroredStrategy)
    if hasattr(tf, 'contrib'):
        _monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
        _monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
        _monkey_patch_dataset_method(tf.contrib.distribute.CollectiveAllReduceStrategy)
    else:
        print('Contrib missing: Skip monkey patch tf.contrib.distribute.*')

def undo_set_up_synthetic_data():
    if False:
        i = 10
        return i + 15
    _undo_monkey_patch_dataset_method(tf.distribute.OneDeviceStrategy)
    _undo_monkey_patch_dataset_method(tf.distribute.MirroredStrategy)
    _undo_monkey_patch_dataset_method(tf.distribute.experimental.MultiWorkerMirroredStrategy)
    if hasattr(tf, 'contrib'):
        _undo_monkey_patch_dataset_method(tf.contrib.distribute.MirroredStrategy)
        _undo_monkey_patch_dataset_method(tf.contrib.distribute.OneDeviceStrategy)
        _undo_monkey_patch_dataset_method(tf.contrib.distribute.CollectiveAllReduceStrategy)
    else:
        print('Contrib missing: Skip remove monkey patch tf.contrib.distribute.*')

def configure_cluster(worker_hosts=None, task_index=-1):
    if False:
        while True:
            i = 10
    'Set multi-worker cluster spec in TF_CONFIG environment variable.\n\n  Args:\n    worker_hosts: comma-separated list of worker ip:port pairs.\n\n  Returns:\n    Number of workers in the cluster.\n  '
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    if tf_config:
        num_workers = len(tf_config['cluster'].get('chief', [])) + len(tf_config['cluster'].get('worker', []))
    elif worker_hosts:
        workers = worker_hosts.split(',')
        num_workers = len(workers)
        if num_workers > 1 and task_index < 0:
            raise ValueError('Must specify task_index when number of workers > 1')
        task_index = 0 if num_workers == 1 else task_index
        os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': workers}, 'task': {'type': 'worker', 'index': task_index}})
    else:
        num_workers = 1
    return num_workers

def get_strategy_scope(strategy):
    if False:
        while True:
            i = 10
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = DummyContextManager()
    return strategy_scope

class DummyContextManager(object):

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        pass