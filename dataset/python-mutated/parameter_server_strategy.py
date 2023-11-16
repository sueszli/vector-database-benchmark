"""Class implementing a multi-worker parameter server tf.distribute strategy."""
import copy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_LOCAL_CPU = '/device:CPU:0'

@tf_export(v1=['distribute.experimental.ParameterServerStrategy'])
class ParameterServerStrategyV1(distribute_lib.StrategyV1):
    """An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two roles: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  ```
  """

    def __init__(self, cluster_resolver=None):
        if False:
            return 10
        'Initializes this strategy with an optional `cluster_resolver`.\n\n    Args:\n      cluster_resolver: Optional\n        `tf.distribute.cluster_resolver.ClusterResolver` object. Defaults to a\n        `tf.distribute.cluster_resolver.TFConfigClusterResolver`.\n    '
        if cluster_resolver is None:
            cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        super(ParameterServerStrategyV1, self).__init__(ParameterServerStrategyExtended(self, cluster_resolver=cluster_resolver))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('ParameterServerStrategy')

    def experimental_distribute_dataset(self, dataset, options=None):
        if False:
            return 10
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).experimental_distribute_dataset(dataset=dataset, options=options)

    def distribute_datasets_from_function(self, dataset_fn, options=None):
        if False:
            print('Hello World!')
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).distribute_datasets_from_function(dataset_fn=dataset_fn, options=options)

    def run(self, fn, args=(), kwargs=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).run(fn, args=args, kwargs=kwargs, options=options)

    def scope(self):
        if False:
            return 10
        self._raise_pss_error_if_eager()
        return super(ParameterServerStrategyV1, self).scope()

    def _raise_pss_error_if_eager(self):
        if False:
            return 10
        if context.executing_eagerly():
            raise NotImplementedError('`tf.compat.v1.distribute.experimental.ParameterServerStrategy` currently only works with the tf.Estimator API')

class ParameterServerStrategyExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of ParameterServerStrategy and CentralStorageStrategy."""

    def __init__(self, container_strategy, cluster_resolver=None, compute_devices=None, parameter_device=None):
        if False:
            while True:
                i = 10
        super(ParameterServerStrategyExtended, self).__init__(container_strategy)
        self._initialize_strategy(cluster_resolver=cluster_resolver, compute_devices=compute_devices, parameter_device=parameter_device)
        self._cross_device_ops = cross_device_ops_lib.ReductionToOneDevice(reduce_to_device=_LOCAL_CPU)

    def _initialize_strategy(self, cluster_resolver=None, compute_devices=None, parameter_device=None):
        if False:
            print('Hello World!')
        if cluster_resolver and cluster_resolver.cluster_spec():
            self._initialize_multi_worker(cluster_resolver)
        else:
            self._initialize_local(compute_devices, parameter_device, cluster_resolver=cluster_resolver)

    def _initialize_multi_worker(self, cluster_resolver):
        if False:
            i = 10
            return i + 15
        "Initialize devices for multiple workers.\n\n    It creates variable devices and compute devices. Variables and operations\n    will be assigned to them respectively. We have one compute device per\n    replica. The variable device is a device function or device string. The\n    default variable device assigns variables to parameter servers in a\n    round-robin fashion.\n\n    Args:\n      cluster_resolver: a descendant of `ClusterResolver` object.\n\n    Raises:\n      ValueError: if the cluster doesn't have ps jobs.\n    "
        if isinstance(cluster_resolver, tfconfig_cluster_resolver.TFConfigClusterResolver):
            num_gpus = context.num_gpus()
        else:
            num_gpus = cluster_resolver.num_accelerators().get('GPU', 0)
        self._num_gpus_per_worker = num_gpus
        cluster_spec = cluster_resolver.cluster_spec()
        task_type = cluster_resolver.task_type
        task_id = cluster_resolver.task_id
        if not task_type or task_id is None:
            raise ValueError('When `cluster_spec` is given, you must also specify `task_type` and `task_id`')
        cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
        assert cluster_spec.as_dict()
        self._worker_device = '/job:%s/task:%d' % (task_type, task_id)
        self._input_host_device = numpy_dataset.SingleDevice(self._worker_device)
        if num_gpus > 0:
            compute_devices = tuple(('%s/device:GPU:%d' % (self._worker_device, i) for i in range(num_gpus)))
        else:
            compute_devices = (self._worker_device,)
        self._compute_devices = [device_util.canonicalize(d) for d in compute_devices]
        num_ps_replicas = len(cluster_spec.as_dict().get('ps', []))
        if num_ps_replicas == 0:
            raise ValueError('The cluster spec needs to have `ps` jobs.')
        self._variable_device = device_setter.replica_device_setter(ps_tasks=num_ps_replicas, worker_device=self._worker_device, merge_devices=True, cluster=cluster_spec)
        self._parameter_devices = tuple(map('/job:ps/task:{}'.format, range(num_ps_replicas)))
        self._default_device = self._worker_device
        self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type, task_id)
        self._cluster_spec = cluster_spec
        self._task_type = task_type
        self._task_id = task_id
        logging.info('Multi-worker ParameterServerStrategy with cluster_spec = %r, task_type = %r, task_id = %r, num_ps_replicas = %r, is_chief = %r, compute_devices = %r, variable_device = %r', cluster_spec.as_dict(), task_type, task_id, num_ps_replicas, self._is_chief, self._compute_devices, self._variable_device)

    def _initialize_local(self, compute_devices, parameter_device, cluster_resolver=None):
        if False:
            while True:
                i = 10
        'Initialize local devices for training.'
        self._worker_device = device_util.canonicalize('/device:CPU:0')
        self._input_host_device = numpy_dataset.SingleDevice(self._worker_device)
        if compute_devices is None:
            if not cluster_resolver:
                num_gpus = context.num_gpus()
            else:
                num_gpus = cluster_resolver.num_accelerators().get('GPU', 0)
            self._num_gpus_per_worker = num_gpus
            compute_devices = device_util.local_devices_from_num_gpus(num_gpus)
        compute_devices = [device_util.canonicalize(d) for d in compute_devices]
        if parameter_device is None:
            if len(compute_devices) == 1:
                parameter_device = compute_devices[0]
            else:
                parameter_device = _LOCAL_CPU
        self._variable_device = parameter_device
        self._compute_devices = compute_devices
        self._parameter_devices = (parameter_device,)
        self._is_chief = True
        self._cluster_spec = None
        self._task_type = None
        self._task_id = None
        logging.info('ParameterServerStrategy (CentralStorageStrategy if you are using a single machine) with compute_devices = %r, variable_device = %r', compute_devices, self._variable_device)

    def _input_workers_with_options(self, options=None):
        if False:
            i = 10
            return i + 15
        if not options or options.experimental_fetch_to_device:
            return input_lib.InputWorkers([(self._worker_device, self._compute_devices)])
        else:
            return input_lib.InputWorkers([(self._worker_device, (self._worker_device,) * len(self._compute_devices))])

    @property
    def _input_workers(self):
        if False:
            print('Hello World!')
        return self._input_workers_with_options()

    def _validate_colocate_with_variable(self, colocate_with_variable):
        if False:
            while True:
                i = 10
        distribute_utils.validate_colocate(colocate_with_variable, self)

    def _experimental_distribute_dataset(self, dataset, options):
        if False:
            i = 10
            return i + 15
        return input_util.get_distributed_dataset(dataset, self._input_workers_with_options(options), self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync, options=options)

    def _make_dataset_iterator(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        return input_lib_v1.DatasetIterator(dataset, self._input_workers, self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync)

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        if False:
            i = 10
            return i + 15
        'Distributes the dataset to each local GPU.'
        if self._cluster_spec:
            input_pipeline_id = multi_worker_util.id_in_cluster(self._cluster_spec, self._task_type, self._task_id)
            num_input_pipelines = multi_worker_util.worker_count(self._cluster_spec, self._task_type)
        else:
            input_pipeline_id = 0
            num_input_pipelines = 1
        input_context = distribute_lib.InputContext(num_input_pipelines=num_input_pipelines, input_pipeline_id=input_pipeline_id, num_replicas_in_sync=self._num_replicas_in_sync)
        return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers, [input_context], self._container_strategy())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        if False:
            while True:
                i = 10
        return numpy_dataset.one_host_numpy_dataset(numpy_input, self._input_host_device, session)

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if False:
            i = 10
            return i + 15
        if self._cluster_spec:
            input_pipeline_id = multi_worker_util.id_in_cluster(self._cluster_spec, self._task_type, self._task_id)
            num_input_pipelines = multi_worker_util.worker_count(self._cluster_spec, self._task_type)
        else:
            input_pipeline_id = 0
            num_input_pipelines = 1
        input_context = distribute_lib.InputContext(num_input_pipelines=num_input_pipelines, input_pipeline_id=input_pipeline_id, num_replicas_in_sync=self._num_replicas_in_sync)
        return input_util.get_distributed_datasets_from_function(dataset_fn, self._input_workers_with_options(options), [input_context], self._container_strategy(), options=options)

    def _experimental_distribute_values_from_function(self, value_fn):
        if False:
            for i in range(10):
                print('nop')
        per_replica_values = []
        for replica_id in range(self._num_replicas_in_sync):
            per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
        return distribute_utils.regroup(per_replica_values, always_wrap=True)

    def _broadcast_to(self, tensor, destinations):
        if False:
            return 10
        if isinstance(tensor, (float, int)):
            return tensor
        if not cross_device_ops_lib.check_destinations(destinations):
            destinations = self._compute_devices
        return self._cross_device_ops.broadcast(tensor, destinations)

    def _allow_variable_partition(self):
        if False:
            i = 10
            return i + 15
        return not context.executing_eagerly()

    def _create_var_creator(self, next_creator, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._num_replicas_in_sync > 1:
            aggregation = kwargs.pop('aggregation', vs.VariableAggregation.NONE)
            if aggregation not in (vs.VariableAggregation.NONE, vs.VariableAggregation.SUM, vs.VariableAggregation.MEAN, vs.VariableAggregation.ONLY_FIRST_REPLICA):
                raise ValueError('Invalid variable aggregation mode: ' + aggregation + ' for variable: ' + kwargs['name'])

            def var_creator(**kwargs):
                if False:
                    while True:
                        i = 10
                'Create an AggregatingVariable and fix up collections.'
                collections = kwargs.pop('collections', None)
                if collections is None:
                    collections = [ops.GraphKeys.GLOBAL_VARIABLES]
                kwargs['collections'] = []
                v = next_creator(**kwargs)
                wrapped = ps_values.AggregatingVariable(self._container_strategy(), v, aggregation)
                if not context.executing_eagerly():
                    g = ops.get_default_graph()
                    if kwargs.get('trainable', True):
                        collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
                        l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
                        if v in l:
                            l.remove(v)
                    g.add_to_collections(collections, wrapped)
                elif ops.GraphKeys.GLOBAL_STEP in collections:
                    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, wrapped)
                return wrapped
            return var_creator
        else:
            return next_creator

    def _create_variable(self, next_creator, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        var_creator = self._create_var_creator(next_creator, **kwargs)
        if 'colocate_with' in kwargs:
            colocate_with = kwargs['colocate_with']
            if isinstance(colocate_with, numpy_dataset.SingleDevice):
                with ops.device(colocate_with.device):
                    return var_creator(**kwargs)
            with ops.device(None):
                with ops.colocate_with(colocate_with):
                    return var_creator(**kwargs)
        with ops.colocate_with(None, ignore_existing=True):
            with ops.device(self._variable_device):
                return var_creator(**kwargs)

    def _call_for_each_replica(self, fn, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return mirrored_run.call_for_each_replica(self._container_strategy(), fn, args, kwargs)

    def _verify_destinations_not_different_worker(self, destinations):
        if False:
            while True:
                i = 10
        if not self._cluster_spec:
            return
        if destinations is None:
            return
        for d in cross_device_ops_lib.get_devices_from(destinations):
            d_spec = tf_device.DeviceSpec.from_string(d)
            if d_spec.job == self._task_type and d_spec.task != self._task_id:
                raise ValueError('Cannot reduce to another worker: %r, current worker is %r' % (d, self._worker_device))

    def _gather_to_implementation(self, value, destinations, axis, options):
        if False:
            return 10
        self._verify_destinations_not_different_worker(destinations)
        if not isinstance(value, values.DistributedValues):
            return value
        return self._cross_device_ops._gather(value, destinations=destinations, axis=axis, options=options)

    def _reduce_to(self, reduce_op, value, destinations, options):
        if False:
            return 10
        self._verify_destinations_not_different_worker(destinations)
        if not isinstance(value, values.DistributedValues):
            return cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, self._num_replicas_in_sync)
        return self._cross_device_ops.reduce(reduce_op, value, destinations=destinations, options=options)

    def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
        if False:
            while True:
                i = 10
        for (_, destinations) in value_destination_pairs:
            self._verify_destinations_not_different_worker(destinations)
        return self._cross_device_ops.batch_reduce(reduce_op, value_destination_pairs, options)

    def _select_single_value(self, structured):
        if False:
            return 10
        'Select any single value in `structured`.'

        def _select_fn(x):
            if False:
                print('Hello World!')
            if isinstance(x, values.Mirrored) or isinstance(x, values.PerReplica):
                return x._primary
            else:
                return x
        return nest.map_structure(_select_fn, structured)

    def _update(self, var, fn, args, kwargs, group):
        if False:
            print('Hello World!')
        if isinstance(var, ps_values.AggregatingVariable):
            var = var.get()
        if not resource_variable_ops.is_resource_variable(var):
            raise ValueError('You can not update `var` %r. It must be a Variable.' % var)
        with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
            result = fn(var, *self._select_single_value(args), **self._select_single_value(kwargs))
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        if False:
            print('Hello World!')
        with ops.device(colocate_with.device), distribute_lib.UpdateContext(colocate_with):
            result = fn(*args, **kwargs)
            if group:
                return result
            else:
                return nest.map_structure(self._local_results, result)

    def value_container(self, val):
        if False:
            i = 10
            return i + 15
        if hasattr(val, '_aggregating_container') and (not isinstance(val, ps_values.AggregatingVariable)):
            wrapper = val._aggregating_container()
            if wrapper is not None:
                return wrapper
        return val

    def read_var(self, var):
        if False:
            i = 10
            return i + 15
        return array_ops.identity(var)

    def _configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        if False:
            while True:
                i = 10
        'Configures the strategy class with `cluster_spec`.\n\n    The strategy object will be re-initialized if `cluster_spec` is passed to\n    `configure` but was not passed when instantiating the strategy.\n\n    Args:\n      session_config: Session config object.\n      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the\n        cluster configurations.\n      task_type: the current task type.\n      task_id: the current task id.\n\n    Raises:\n      ValueError: if `cluster_spec` is given but `task_type` or `task_id` is\n        not.\n    '
        if cluster_spec:
            cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec), task_type=task_type, task_id=task_id, num_accelerators={'GPU': self._num_gpus_per_worker})
            self._initialize_multi_worker(cluster_resolver)
        if session_config:
            session_config.CopyFrom(self._update_config_proto(session_config))

    def _update_config_proto(self, config_proto):
        if False:
            print('Hello World!')
        updated_config = copy.deepcopy(config_proto)
        if not self._cluster_spec:
            updated_config.isolate_session_state = True
            return updated_config
        updated_config.isolate_session_state = False
        assert self._task_type
        assert self._task_id is not None
        del updated_config.device_filters[:]
        if self._task_type in ['chief', 'worker']:
            updated_config.device_filters.extend(['/job:%s/task:%d' % (self._task_type, self._task_id), '/job:ps'])
        elif self._task_type == 'evaluator':
            updated_config.device_filters.append('/job:%s/task:%d' % (self._task_type, self._task_id))
        return updated_config

    def _in_multi_worker_mode(self):
        if False:
            return 10
        'Whether this strategy indicates working in multi-worker settings.'
        return self._cluster_spec is not None

    @property
    def _num_replicas_in_sync(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._compute_devices)

    @property
    def worker_devices(self):
        if False:
            print('Hello World!')
        return self._compute_devices

    @property
    def worker_devices_by_replica(self):
        if False:
            print('Hello World!')
        return [[d] for d in self._compute_devices]

    @property
    def parameter_devices(self):
        if False:
            for i in range(10):
                print('nop')
        return self._parameter_devices

    def non_slot_devices(self, var_list):
        if False:
            while True:
                i = 10
        return min(var_list, key=lambda x: x.name)

    @property
    def experimental_between_graph(self):
        if False:
            return 10
        return True

    @property
    def experimental_should_init(self):
        if False:
            return 10
        return self._is_chief

    @property
    def should_checkpoint(self):
        if False:
            return 10
        return self._is_chief

    @property
    def should_save_summary(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_chief

    @property
    def _global_batch_size(self):
        if False:
            return 10
        '`make_dataset_iterator` and `make_numpy_iterator` use global batch size.\n\n    `make_input_fn_iterator` assumes per-replica batching.\n\n    Returns:\n      Boolean.\n    '
        return True

    def _get_local_replica_id(self, replica_id_in_sync_group):
        if False:
            while True:
                i = 10
        return replica_id_in_sync_group

    def _get_replica_id_in_sync_group(self, replica_id):
        if False:
            print('Hello World!')
        return replica_id