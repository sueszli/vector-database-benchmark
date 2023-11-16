"""Distributed placement strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import contextlib
from absl import logging
from adanet import tf_compat
from adanet.distributed.devices import _OpNameHashStrategy
import numpy as np
import six

@six.add_metaclass(abc.ABCMeta)
class PlacementStrategy(object):
    """Abstract placement strategy for distributed training.

  Given a cluster of workers, the placement strategy determines which subgraph
  each worker constructs.
  """

    @property
    def config(self):
        if False:
            i = 10
            return i + 15
        "Returns this strategy's configuration.\n\n    Returns:\n      The :class:`tf.estimator.RunConfig` instance that defines the cluster.\n    "
        return self._config

    @config.setter
    def config(self, config):
        if False:
            print('Hello World!')
        'Configures the placement strategy with the given cluster description.\n\n    Args:\n      config: A :class:`tf.estimator.RunConfig` instance that defines the\n        cluster.\n    '
        self._config = config

    @abc.abstractmethod
    def should_build_ensemble(self, num_subnetworks):
        if False:
            i = 10
            return i + 15
        'Whether to build the ensemble on the current worker.\n\n    Args:\n      num_subnetworks: Integer number of subnetworks to train in the current\n        iteration.\n\n    Returns:\n      Boolean whether to build the ensemble on the current worker.\n    '

    @abc.abstractmethod
    def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
        if False:
            i = 10
            return i + 15
        "Whether to build the given subnetwork on the current worker.\n\n    Args:\n      num_subnetworks: Integer number of subnetworks to train in the current\n        iteration.\n      subnetwork_index: Integer index of the subnetwork in the list of the\n        current iteration's subnetworks.\n\n    Returns:\n      Boolean whether to build the given subnetwork on the current worker.\n    "

    @abc.abstractmethod
    def should_train_subnetworks(self, num_subnetworks):
        if False:
            for i in range(10):
                print('nop')
        'Whether to train subnetworks on the current worker.\n\n    Args:\n      num_subnetworks: Integer number of subnetworks to train in the current\n        iteration.\n\n    Returns:\n      Boolean whether to train subnetworks on the current worker.\n    '

    @abc.abstractmethod
    @contextlib.contextmanager
    def subnetwork_devices(self, num_subnetworks, subnetwork_index):
        if False:
            print('Hello World!')
        'A context for assigning subnetwork ops to devices.'

class ReplicationStrategy(PlacementStrategy):
    """A simple strategy that replicates the same graph on every worker.

  This strategy does not scale well as the number of subnetworks and workers
  increases. For :math:`m` workers, :math:`n` parameter servers, and :math:`k`
  subnetworks, this strategy will scale with :math:`O(m)` training speedup,
  :math:`O(m*n*k)` variable fetches from parameter servers, and :math:`O(k)`
  memory required per worker. Additionally there will be :math:`O(m)` stale
  gradients per subnetwork when training with asynchronous SGD.

  Returns:
    A :class:`ReplicationStrategy` instance for the current cluster.
  """

    def should_build_ensemble(self, num_subnetworks):
        if False:
            for i in range(10):
                print('nop')
        return True

    def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
        if False:
            return 10
        return True

    def should_train_subnetworks(self, num_subnetworks):
        if False:
            i = 10
            return i + 15
        return True

    @contextlib.contextmanager
    def subnetwork_devices(self, num_subnetworks, subnetwork_index):
        if False:
            return 10
        yield

class RoundRobinStrategy(PlacementStrategy):
    """A strategy that round-robin assigns subgraphs to specific workers.

  Specifically, it selects dedicated workers to only train ensemble variables,
  and round-robin assigns subnetworks to dedicated subnetwork-training workers.

  Unlike :class:`ReplicationStrategy`, this strategy scales better with the
  number of subnetworks, workers, and parameter servers. For :math:`m` workers,
  :math:`n` parameter servers, and :math:`k` subnetworks, this strategy will
  scale with :math:`O(m/k)` training speedup, :math:`O(m*n/k)` variable fetches
  from parameter servers, and :math:`O(1)` memory required per worker.
  Additionally, there will only be :math:`O(m/k)` stale gradients per subnetwork
  when training with asynchronous SGD, which reduces training instability versus
  :class:`ReplicationStrategy`.

  When there are more workers than subnetworks, this strategy assigns
  subnetworks to workers modulo the number of subnetworks.

  Conversely, when there are more subnetworks than workers, this round robin
  assigns subnetworks modulo the number of workers. So certain workers may end
  up training more than one subnetwork.

  This strategy gracefully handles scenarios when the number of subnetworks
  does not perfectly divide the number of workers and vice-versa. It also
  supports different numbers of subnetworks at different iterations, and
  reloading training with a resized cluster.

  Args:
    drop_remainder: Bool whether to drop remaining subnetworks that haven't been
      assigned to a worker in the remainder after perfect division of workers by
      the current iteration's num_subnetworks + 1. When :code:`True`, each subnetwork
      worker will only train a single subnetwork, and subnetworks that have not
      been assigned to assigned to a worker are dropped. NOTE: This can result
      in subnetworks not being assigned to any worker when
      num_workers < num_subnetworks + 1. When :code:`False`, remaining subnetworks
      during the round-robin assignment will be placed on workers that already
      have a subnetwork.

  Returns:
    A :class:`RoundRobinStrategy` instance for the current cluster.
  """

    def __init__(self, drop_remainder=False, dedicate_parameter_servers=True):
        if False:
            print('Hello World!')
        self._drop_remainder = drop_remainder
        self._dedicate_parameter_servers = dedicate_parameter_servers

    @property
    def _num_workers(self):
        if False:
            i = 10
            return i + 15
        return self.config.num_worker_replicas

    @property
    def _worker_index(self):
        if False:
            while True:
                i = 10
        return self.config.global_id_in_cluster or 0

    def _worker_task(self, num_subnetworks):
        if False:
            while True:
                i = 10
        'Returns the worker index modulo the number of subnetworks.'
        if self._drop_remainder and self._num_workers > 1 and (num_subnetworks > self._num_workers):
            logging.log_first_n(logging.WARNING, 'With drop_remainer=True, %s workers and %s subnetworks, the last %s subnetworks will be dropped and will not be trained', 1, self._num_workers, num_subnetworks, num_subnetworks - self._num_workers - 1)
        return self._worker_index % (num_subnetworks + 1)

    def should_build_ensemble(self, num_subnetworks):
        if False:
            for i in range(10):
                print('nop')
        if num_subnetworks == 1:
            return True
        worker_task = self._worker_task(num_subnetworks)
        return worker_task == 0

    def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
        if False:
            while True:
                i = 10
        if num_subnetworks == 1:
            return True
        worker_task = self._worker_task(num_subnetworks)
        if worker_task == 0:
            return True
        subnetwork_worker_index = worker_task - 1
        if self._drop_remainder:
            return subnetwork_worker_index == subnetwork_index
        workers_per_subnetwork = self._num_workers // (num_subnetworks + 1)
        if self._num_workers % (num_subnetworks + 1) == 0:
            num_subnetwork_workers = num_subnetworks
        elif self._worker_index >= workers_per_subnetwork * (num_subnetworks + 1):
            num_subnetwork_workers = self._num_workers % (num_subnetworks + 1) - 1
        else:
            num_subnetwork_workers = num_subnetworks
        return subnetwork_worker_index == subnetwork_index % num_subnetwork_workers

    def should_train_subnetworks(self, num_subnetworks):
        if False:
            for i in range(10):
                print('nop')
        if num_subnetworks == 1 or self._num_workers == 1:
            return True
        return not self.should_build_ensemble(num_subnetworks)

    @contextlib.contextmanager
    def subnetwork_devices(self, num_subnetworks, subnetwork_index):
        if False:
            return 10
        if not self._dedicate_parameter_servers:
            yield
            return
        num_ps_replicas = self.config.num_ps_replicas
        ps_numbers = np.array(range(num_ps_replicas))
        subnetwork_group = subnetwork_index
        if num_ps_replicas > 0 and num_subnetworks > num_ps_replicas:
            subnetwork_group = subnetwork_index % num_ps_replicas
        ps_group = np.array_split(ps_numbers, num_subnetworks)[subnetwork_group]
        ps_strategy = _OpNameHashStrategy(len(ps_group))

        def device_fn(op):
            if False:
                return 10
            "Assigns variables to a subnetwork's dedicated parameter servers."
            from tensorflow.core.framework import node_def_pb2
            node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
            from tensorflow.python.training import device_setter
            if num_ps_replicas > 0 and node_def.op in device_setter.STANDARD_PS_OPS:
                return '/job:ps/task:{}'.format(ps_group[0] + ps_strategy(op))
            return op.device
        with tf_compat.v1.device(device_fn):
            yield