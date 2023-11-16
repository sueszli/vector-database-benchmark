"""Class MirroredStrategy implementing tf.distribute.Strategy."""
import copy
from tensorflow.python import tf2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

def _is_device_list_single_worker(devices):
    if False:
        i = 10
        return i + 15
    'Checks whether the devices list is for single or multi-worker.\n\n  Args:\n    devices: a list of device strings or tf.config.LogicalDevice objects, for\n      either local or for remote devices.\n\n  Returns:\n    a boolean indicating whether these device strings are for local or for\n    remote.\n\n  Raises:\n    ValueError: if device strings are not consistent.\n  '
    specs = []
    for d in devices:
        name = d.name if isinstance(d, context.LogicalDevice) else d
        specs.append(tf_device.DeviceSpec.from_string(name))
    num_workers = len({(d.job, d.task, d.replica) for d in specs})
    all_local = all((d.job in (None, 'localhost') for d in specs))
    any_local = any((d.job in (None, 'localhost') for d in specs))
    if any_local and (not all_local):
        raise ValueError("Local device should have only 'localhost' in the job field in device string. E.g. 'job:localhost' in /job:localhost/replica:0/task:0/device:CPU:0Devices cannot have mixed list of device strings containing both localhost and other job types such as worker, ps etc. ")
    if num_workers == 1 and (not all_local):
        if any((d.task is None for d in specs)):
            raise ValueError("Remote device string must have task specified.E.g. 'task:0' in /job:worker/replica:0/task:0/device:CPU:0")
    return num_workers == 1

def _cluster_spec_to_device_list(cluster_spec, num_gpus_per_worker):
    if False:
        print('Hello World!')
    'Returns a device list given a cluster spec.'
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    devices = []
    for task_type in ('chief', 'worker'):
        for task_id in range(len(cluster_spec.as_dict().get(task_type, []))):
            if num_gpus_per_worker == 0:
                devices.append('/job:%s/task:%d/device:CPU:0' % (task_type, task_id))
            else:
                devices.extend(['/job:%s/task:%d/device:GPU:%i' % (task_type, task_id, gpu_id) for gpu_id in range(num_gpus_per_worker)])
    return devices

def _group_device_list(devices):
    if False:
        print('Hello World!')
    'Groups the devices list by task_type and task_id.\n\n  Args:\n    devices: a list of device strings for remote devices.\n\n  Returns:\n    a dict of list of device strings mapping from task_type to a list of devices\n    for the task_type in the ascending order of task_id.\n  '
    assert not _is_device_list_single_worker(devices)
    device_dict = {}
    for d in devices:
        d_spec = tf_device.DeviceSpec.from_string(d)
        if d_spec.job not in device_dict:
            device_dict[d_spec.job] = []
        while len(device_dict[d_spec.job]) <= d_spec.task:
            device_dict[d_spec.job].append([])
        device_dict[d_spec.job][d_spec.task].append(d)
    return device_dict

def _is_gpu_device(device):
    if False:
        while True:
            i = 10
    return tf_device.DeviceSpec.from_string(device).device_type == 'GPU'

def _infer_num_gpus_per_worker(devices):
    if False:
        while True:
            i = 10
    'Infers the number of GPUs on each worker.\n\n  Currently to make multi-worker cross device ops work, we need all workers to\n  have the same number of GPUs.\n\n  Args:\n    devices: a list of device strings, can be either local devices or remote\n      devices.\n\n  Returns:\n    number of GPUs per worker.\n\n  Raises:\n    ValueError if workers have different number of GPUs or GPU indices are not\n    consecutive and starting from 0.\n  '
    if _is_device_list_single_worker(devices):
        return sum((1 for d in devices if _is_gpu_device(d)))
    else:
        device_dict = _group_device_list(devices)
        num_gpus = None
        for (_, devices_in_task) in device_dict.items():
            for device_in_task in devices_in_task:
                if num_gpus is None:
                    num_gpus = sum((1 for d in device_in_task if _is_gpu_device(d)))
                elif num_gpus != sum((1 for d in device_in_task if _is_gpu_device(d))):
                    raise ValueError('All workers should have the same number of GPUs.')
                for d in device_in_task:
                    d_spec = tf_device.DeviceSpec.from_string(d)
                    if d_spec.device_type == 'GPU' and d_spec.device_index >= num_gpus:
                        raise ValueError('GPU `device_index` on a worker should be consecutive and start from 0.')
        return num_gpus

def all_local_devices(num_gpus=None):
    if False:
        return 10
    devices = config.list_logical_devices('GPU')
    if num_gpus is not None:
        devices = devices[:num_gpus]
    return devices or config.list_logical_devices('CPU')

def all_devices():
    if False:
        while True:
            i = 10
    devices = []
    tfconfig = tfconfig_cluster_resolver.TFConfigClusterResolver()
    if tfconfig.cluster_spec().as_dict():
        devices = _cluster_spec_to_device_list(tfconfig.cluster_spec(), context.num_gpus())
    return devices if devices else all_local_devices()

@tf_export('distribute.MirroredStrategy', v1=[])
class MirroredStrategy(distribute_lib.Strategy):
    """Synchronous training across multiple replicas on one machine.

  This strategy is typically used for training on one
  machine with multiple GPUs. For TPUs, use
  `tf.distribute.TPUStrategy`. To use `MirroredStrategy` with multiple workers,
  please refer to `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

  For example, a variable created under a `MirroredStrategy` is a
  `MirroredVariable`. If no devices are specified in the constructor argument of
  the strategy then it will use all the available GPUs. If no GPUs are found, it
  will use the available CPUs. Note that TensorFlow treats all CPUs on a
  machine as a single device, and uses threads internally for parallelism.

  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   x = tf.Variable(1.)
  >>> x
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
  }

  While using distribution strategies, all the variable creation should be done
  within the strategy's scope. This will replicate the variables across all the
  replicas and keep them in sync using an all-reduce algorithm.

  Variables created inside a `MirroredStrategy` which is wrapped with a
  `tf.function` are still `MirroredVariables`.

  >>> x = []
  >>> @tf.function  # Wrap the function with tf.function.
  ... def create_variable():
  ...   if not x:
  ...     x.append(tf.Variable(1.))
  ...   return x[0]
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   _ = create_variable()
  ...   print(x[0])
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
  }

  `experimental_distribute_dataset` can be used to distribute the dataset across
  the replicas when writing your own training loop. If you are using `.fit` and
  `.compile` methods available in `tf.keras`, then `tf.keras` will handle the
  distribution for you.

  For example:

  ```python
  my_strategy = tf.distribute.MirroredStrategy()
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result

      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_result, axis=None)
      return total_result

    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```

  Args:
    devices: a list of device strings such as `['/gpu:0', '/gpu:1']`.  If
      `None`, all available GPUs are used. If no GPUs are found, CPU is used.
    cross_device_ops: optional, a descendant of `CrossDeviceOps`. If this is not
      set, `NcclAllReduce()` will be used by default.  One would customize this
      if NCCL isn't available or if a special implementation that exploits
      the particular hardware is available.
  """
    _collective_key_base = 0

    def __init__(self, devices=None, cross_device_ops=None):
        if False:
            i = 10
            return i + 15
        extended = MirroredExtended(self, devices=devices, cross_device_ops=cross_device_ops)
        super(MirroredStrategy, self).__init__(extended)
        distribute_lib.distribution_strategy_gauge.get_cell('V2').set('MirroredStrategy')

@tf_export(v1=['distribute.MirroredStrategy'])
class MirroredStrategyV1(distribute_lib.StrategyV1):
    __doc__ = MirroredStrategy.__doc__
    _collective_key_base = 0

    def __init__(self, devices=None, cross_device_ops=None):
        if False:
            return 10
        extended = MirroredExtended(self, devices=devices, cross_device_ops=cross_device_ops)
        super(MirroredStrategyV1, self).__init__(extended)
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('MirroredStrategy')

class MirroredExtended(distribute_lib.StrategyExtendedV1):
    """Implementation of MirroredStrategy."""

    def __init__(self, container_strategy, devices=None, cross_device_ops=None):
        if False:
            i = 10
            return i + 15
        super(MirroredExtended, self).__init__(container_strategy)
        if context.executing_eagerly():
            if devices and (not _is_device_list_single_worker(devices)):
                raise RuntimeError('In-graph multi-worker training with `MirroredStrategy` is not supported in eager mode.')
            else:
                if tfconfig_cluster_resolver.TFConfigClusterResolver().cluster_spec().as_dict():
                    logging.info('Initializing local devices since in-graph multi-worker training with `MirroredStrategy` is not supported in eager mode. TF_CONFIG will be ignored when when initializing `MirroredStrategy`.')
                devices = devices or all_local_devices()
        else:
            devices = devices or all_devices()
        assert devices, 'Got an empty `devices` list and unable to recognize any local devices.'
        self._collective_key_base = container_strategy._collective_key_base
        self._communication_options = collective_util.Options(implementation=collective_util.CommunicationImplementation.NCCL)
        self._cross_device_ops = cross_device_ops
        self._initialize_strategy(devices)
        if ops.executing_eagerly_outside_functions():
            self.experimental_enable_get_next_as_optional = True
        self._use_var_policy = False

    def _use_merge_call(self):
        if False:
            while True:
                i = 10
        return not control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()) or not all([_is_gpu_device(d) for d in self._devices])

    def _initialize_strategy(self, devices):
        if False:
            for i in range(10):
                print('nop')
        assert devices, 'Must specify at least one device.'
        devices = tuple((device_util.resolve(d) for d in devices))
        assert len(set(devices)) == len(devices), 'No duplicates allowed in `devices` argument: %s' % (devices,)
        self._initialize_single_worker(devices)
        self._collective_ops = self._make_collective_ops_with_fallbacks()
        if not self._cross_device_ops:
            self._cross_device_ops = self._collective_ops

    def _make_collective_ops_with_fallbacks(self):
        if False:
            return 10
        self._collective_keys = cross_device_utils.CollectiveKeys(group_key_start=1 + self._collective_key_base)
        if not ops.executing_eagerly_outside_functions() and any(('gpu' not in d.lower() for d in self._devices)):
            return cross_device_ops_lib.ReductionToOneDevice()
        if any(('cpu' in d.lower() for d in self._devices)) and any(('gpu' in d.lower() for d in self._devices)):
            return cross_device_ops_lib.ReductionToOneDevice()
        if all(('cpu' in d.lower() for d in self._devices)):
            self._communication_options = collective_util.Options(implementation=collective_util.CommunicationImplementation.RING)
        else:
            physical_gpus = context.context().list_physical_devices(device_type='GPU')
            logical_gpus = context.context().list_logical_devices(device_type='GPU')
            if len(physical_gpus) < len(logical_gpus):
                self._communication_options = collective_util.Options(implementation=collective_util.CommunicationImplementation.RING)
        return cross_device_ops_lib.CollectiveAllReduce(devices=self._devices, group_size=len(self._devices), options=self._communication_options, collective_keys=self._collective_keys)

    def _initialize_single_worker(self, devices):
        if False:
            print('Hello World!')
        'Initializes the object for single-worker training.'
        self._devices = tuple((device_util.canonicalize(d) for d in devices))
        self._input_workers_devices = ((device_util.canonicalize('/device:CPU:0', devices[0]), devices),)
        self._host_input_device = numpy_dataset.SingleDevice(self._input_workers_devices[0][0])
        device_spec = tf_device.DeviceSpec.from_string(self._input_workers_devices[0][0])
        if device_spec.job is not None and device_spec.job != 'localhost':
            self._default_device = '/job:%s/replica:%d/task:%d' % (device_spec.job, device_spec.replica, device_spec.task)
        logging.info('Using MirroredStrategy with devices %r', devices)

    def _initialize_multi_worker(self, devices):
        if False:
            while True:
                i = 10
        'Initializes the object for multi-worker training.'
        device_dict = _group_device_list(devices)
        workers = []
        worker_devices = []
        for job in ('chief', 'worker'):
            for task in range(len(device_dict.get(job, []))):
                worker = '/job:%s/task:%d' % (job, task)
                workers.append(worker)
                worker_devices.append((worker, device_dict[job][task]))
        self._default_device = workers[0]
        self._host_input_device = numpy_dataset.SingleDevice(workers[0])
        self._devices = tuple(devices)
        self._input_workers_devices = worker_devices
        self._is_multi_worker_training = True
        if len(workers) > 1:
            if not isinstance(self._cross_device_ops, cross_device_ops_lib.ReductionToOneDevice) or self._cross_device_ops._num_between_graph_workers > 1:
                raise ValueError('In-graph multi-worker training with `MirroredStrategy` is not supported.')
            self._inferred_cross_device_ops = self._cross_device_ops
        else:
            self._inferred_cross_device_ops = cross_device_ops_lib.NcclAllReduce()
        logging.info('Using MirroredStrategy with remote devices %r', devices)

    def _input_workers_with_options(self, options=None):
        if False:
            for i in range(10):
                print('nop')
        if not options:
            return input_lib.InputWorkers(self._input_workers_devices)
        if options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            if options.experimental_place_dataset_on_device:
                self._input_workers_devices = tuple(((device_util.canonicalize(d, d), (d,)) for d in self._devices))
            else:
                self._input_workers_devices = tuple(((device_util.canonicalize('/device:CPU:0', d), (d,)) for d in self._devices))
            return input_lib.InputWorkers(self._input_workers_devices)
        elif not options.experimental_fetch_to_device:
            return input_lib.InputWorkers([(host_device, (host_device,) * len(compute_devices)) for (host_device, compute_devices) in self._input_workers_devices])
        else:
            return input_lib.InputWorkers(self._input_workers_devices)

    @property
    def _input_workers(self):
        if False:
            while True:
                i = 10
        return self._input_workers_with_options()

    def _get_variable_creator_initial_value(self, replica_id, device, primary_var, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return the initial value for variables on a replica.'
        if replica_id == 0:
            return kwargs['initial_value']
        else:
            assert primary_var is not None
            assert device is not None
            assert kwargs is not None

            def initial_value_fn():
                if False:
                    print('Hello World!')
                if context.executing_eagerly() or ops.inside_function():
                    init_value = primary_var.value()
                    return array_ops.identity(init_value)
                else:
                    with ops.device(device):
                        init_value = primary_var.initial_value
                        return array_ops.identity(init_value)
            return initial_value_fn

    def _create_variable(self, next_creator, **kwargs):
        if False:
            while True:
                i = 10
        'Create a mirrored variable. See `DistributionStrategy.scope`.'
        colocate_with = kwargs.pop('colocate_with', None)
        if colocate_with is None:
            devices = self._devices
        elif isinstance(colocate_with, numpy_dataset.SingleDevice):
            with ops.device(colocate_with.device):
                return next_creator(**kwargs)
        else:
            devices = colocate_with._devices

        def _real_mirrored_creator(**kwargs):
            if False:
                i = 10
                return i + 15
            value_list = []
            for (i, d) in enumerate(devices):
                with ops.device(d):
                    kwargs['initial_value'] = self._get_variable_creator_initial_value(replica_id=i, device=d, primary_var=value_list[0] if value_list else None, **kwargs)
                    if i > 0:
                        var0name = value_list[0].name.split(':')[0]
                        kwargs['name'] = '%s/replica_%d/' % (var0name, i)
                    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                        with record.stop_recording():
                            v = next_creator(**kwargs)
                    assert not isinstance(v, values.DistributedVariable)
                    value_list.append(v)
            return value_list
        return distribute_utils.create_mirrored_variable(self._container_strategy(), _real_mirrored_creator, distribute_utils.VARIABLE_CLASS_MAPPING, distribute_utils.VARIABLE_POLICY_MAPPING, **kwargs)

    def _validate_colocate_with_variable(self, colocate_with_variable):
        if False:
            print('Hello World!')
        distribute_utils.validate_colocate_distributed_variable(colocate_with_variable, self)

    def _make_dataset_iterator(self, dataset):
        if False:
            return 10
        return input_lib_v1.DatasetIterator(dataset, self._input_workers, self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync)

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        if False:
            while True:
                i = 10
        input_contexts = []
        num_workers = self._input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers, input_contexts, self._container_strategy())

    def _experimental_distribute_dataset(self, dataset, options):
        if False:
            for i in range(10):
                print('nop')
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `distribute_datasets_from_function`.')
        return input_util.get_distributed_dataset(dataset, self._input_workers_with_options(options), self._container_strategy(), num_replicas_in_sync=self._num_replicas_in_sync, options=options)

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        if False:
            for i in range(10):
                print('nop')
        return numpy_dataset.one_host_numpy_dataset(numpy_input, self._host_input_device, session)

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if False:
            for i in range(10):
                print('nop')
        input_workers = self._input_workers_with_options(options)
        input_contexts = []
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=self._num_replicas_in_sync))
        return input_util.get_distributed_datasets_from_function(dataset_fn, input_workers, input_contexts, self._container_strategy(), options)

    def _experimental_distribute_values_from_function(self, value_fn):
        if False:
            while True:
                i = 10
        per_replica_values = []
        for replica_id in range(self._num_replicas_in_sync):
            per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
        return distribute_utils.regroup(per_replica_values, always_wrap=True)

    def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values=None):
        if False:
            while True:
                i = 10
        if initial_loop_values is None:
            initial_loop_values = {}
        initial_loop_values = nest.flatten(initial_loop_values)
        ctx = input_lib.MultiStepContext()

        def body(i, *args):
            if False:
                i = 10
                return i + 15
            'A wrapper around `fn` to create the while loop body.'
            del args
            fn_result = fn(ctx, iterator.get_next())
            for (name, output) in ctx.last_step_outputs.items():
                ctx.last_step_outputs[name] = self._local_results(output)
            flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
            with ops.control_dependencies([fn_result]):
                return [i + 1] + flat_last_step_outputs
        self._outer_control_flow_context = ops.get_default_graph()._get_control_flow_context()
        cond = lambda i, *args: i < iterations
        i = constant_op.constant(0)
        loop_result = while_loop.while_loop(cond, body, [i] + initial_loop_values, name='', parallel_iterations=1, back_prop=False, swap_memory=False, return_same_structure=True)
        del self._outer_control_flow_context
        ctx.run_op = control_flow_ops.group(loop_result)
        last_step_tensor_outputs = loop_result[1:]
        last_step_tensor_outputs_dict = nest.pack_sequence_as(ctx.last_step_outputs, last_step_tensor_outputs)
        for (name, reduce_op) in ctx._last_step_outputs_reduce_ops.items():
            output = last_step_tensor_outputs_dict[name]
            if reduce_op is None:
                last_step_tensor_outputs_dict[name] = distribute_utils.regroup(output)
            else:
                assert len(output) == 1
                last_step_tensor_outputs_dict[name] = output[0]
        ctx._set_last_step_outputs(last_step_tensor_outputs_dict)
        return ctx

    def _broadcast_to(self, tensor, destinations):
        if False:
            while True:
                i = 10
        if isinstance(tensor, (float, int)):
            return tensor
        if not destinations:
            destinations = self._devices
        return self._get_cross_device_ops(tensor).broadcast(tensor, destinations)

    def _call_for_each_replica(self, fn, args, kwargs):
        if False:
            i = 10
            return i + 15
        return mirrored_run.call_for_each_replica(self._container_strategy(), fn, args, kwargs)

    def _configure(self, session_config=None, cluster_spec=None, task_type=None, task_id=None):
        if False:
            return 10
        del task_type, task_id
        if session_config:
            session_config.CopyFrom(self._update_config_proto(session_config))
        if cluster_spec:
            num_gpus_per_worker = _infer_num_gpus_per_worker(self._devices)
            multi_worker_devices = _cluster_spec_to_device_list(cluster_spec, num_gpus_per_worker)
            self._initialize_multi_worker(multi_worker_devices)

    def _update_config_proto(self, config_proto):
        if False:
            print('Hello World!')
        updated_config = copy.deepcopy(config_proto)
        updated_config.isolate_session_state = True
        return updated_config

    def _get_cross_device_ops(self, value):
        if False:
            print('Hello World!')
        if not self._use_merge_call():
            if not isinstance(self._cross_device_ops, cross_device_ops_lib.CollectiveAllReduce):
                logging.warning('Under XLA context, MirroredStrategy uses CollectiveAllReduce op. Although %r is provided to initialize MirroredStrategy, it is ignored in XLA. Please use CollectiveAllReduce(or default option) in the future, since other cross device ops are not well supported on XLA.', self._cross_device_ops)
            return self._collective_ops
        if isinstance(value, values.DistributedValues):
            value_int32 = True in {dtypes.as_dtype(v.dtype) == dtypes.int32 for v in value.values}
        else:
            value_int32 = dtypes.as_dtype(value.dtype) == dtypes.int32
        if value_int32:
            return cross_device_ops_lib.ReductionToOneDevice()
        else:
            return self._cross_device_ops

    def _gather_to_implementation(self, value, destinations, axis, options):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, values.DistributedValues):
            return value
        return self._get_cross_device_ops(value)._gather(value, destinations=destinations, axis=axis, options=self._communication_options.merge(options))

    def _reduce_to(self, reduce_op, value, destinations, options):
        if False:
            i = 10
            return i + 15
        if distribute_utils.is_mirrored(value) and reduce_op == reduce_util.ReduceOp.MEAN:
            return value
        assert not distribute_utils.is_mirrored(value)

        def get_values(value):
            if False:
                return 10
            if not isinstance(value, values.DistributedValues):
                return cross_device_ops_lib.reduce_non_distributed_value(reduce_op, value, destinations, self._num_replicas_in_sync)
            if self._use_merge_call() and (not cross_device_ops_lib._devices_match(value, destinations) or any(('cpu' in d.lower() for d in cross_device_ops_lib.get_devices_from(destinations)))):
                return cross_device_ops_lib.ReductionToOneDevice().reduce(reduce_op, value, destinations)
            return self._get_cross_device_ops(value).reduce(reduce_op, value, destinations=destinations, options=self._communication_options.merge(options))
        return nest.map_structure(get_values, value)

    def _batch_reduce_to(self, reduce_op, value_destination_pairs, options):
        if False:
            for i in range(10):
                print('nop')
        cross_device_ops = None
        for (value, _) in value_destination_pairs:
            if cross_device_ops is None:
                cross_device_ops = self._get_cross_device_ops(value)
            elif cross_device_ops is not self._get_cross_device_ops(value):
                raise ValueError('Inputs to batch_reduce_to must be either all on the host or all on the compute devices.')
        return cross_device_ops.batch_reduce(reduce_op, value_destination_pairs, options=self._communication_options.merge(options))

    def _update(self, var, fn, args, kwargs, group):
        if False:
            return 10
        assert isinstance(var, values.DistributedVariable)
        updates = []
        for (i, v) in enumerate(var.values):
            name = 'update_%d' % i
            with ops.device(v.device), distribute_lib.UpdateContext(i), ops.name_scope(name):
                updates.append(fn(v, *distribute_utils.select_replica(i, args), **distribute_utils.select_replica(i, kwargs)))
        return distribute_utils.update_regroup(self, updates, group)

    def _replica_ctx_all_reduce(self, reduce_op, value, options=None):
        if False:
            while True:
                i = 10
        'Implements `StrategyExtendedV2._replica_ctx_all_reduce`.'
        if options is None:
            options = collective_util.Options()
        if context.executing_eagerly() or not tf2.enabled() or self._use_merge_call():
            return super()._replica_ctx_all_reduce(reduce_op, value, options)
        replica_context = distribute_lib.get_replica_context()
        assert replica_context, '`StrategyExtended._replica_ctx_all_reduce` must be called in a replica context'
        return self._get_cross_device_ops(value)._all_reduce(reduce_op, value, replica_context._replica_id, options)

    def _replica_ctx_update(self, var, fn, args, kwargs, group):
        if False:
            while True:
                i = 10
        if self._use_merge_call():
            return super()._replica_ctx_update(var, fn, args, kwargs, group)
        replica_context = distribute_lib.get_replica_context()
        assert replica_context
        replica_id = values_util.get_current_replica_id_as_int()
        name = 'update_%d' % replica_id
        if isinstance(var, values.DistributedVariable):
            var = var._get_replica(replica_id)
        with ops.device(var.device), ops.name_scope(name):
            result = fn(var, *args, **kwargs)
        return result

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        if False:
            return 10
        assert isinstance(colocate_with, tuple)
        updates = []
        for (i, d) in enumerate(colocate_with):
            name = 'update_%d' % i
            with ops.device(d), distribute_lib.UpdateContext(i), ops.name_scope(name):
                updates.append(fn(*distribute_utils.select_replica(i, args), **distribute_utils.select_replica(i, kwargs)))
        return distribute_utils.update_regroup(self, updates, group)

    def read_var(self, replica_local_var):
        if False:
            while True:
                i = 10
        'Read the aggregate value of a replica-local variable.'
        if distribute_utils.is_sync_on_read(replica_local_var):
            return replica_local_var._get_cross_replica()
        assert distribute_utils.is_mirrored(replica_local_var)
        return array_ops.identity(replica_local_var._get())

    def value_container(self, val):
        if False:
            for i in range(10):
                print('nop')
        return distribute_utils.value_container(val)

    @property
    def _num_replicas_in_sync(self):
        if False:
            print('Hello World!')
        return len(self._devices)

    @property
    def worker_devices(self):
        if False:
            for i in range(10):
                print('nop')
        return self._devices

    @property
    def worker_devices_by_replica(self):
        if False:
            while True:
                i = 10
        return [[d] for d in self._devices]

    @property
    def parameter_devices(self):
        if False:
            return 10
        return self.worker_devices

    @property
    def experimental_between_graph(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def experimental_should_init(self):
        if False:
            i = 10
            return i + 15
        return True

    @property
    def should_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def should_save_summary(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def non_slot_devices(self, var_list):
        if False:
            i = 10
            return i + 15
        del var_list
        return self._devices

    @property
    def _global_batch_size(self):
        if False:
            i = 10
            return i + 15
        '`make_dataset_iterator` and `make_numpy_iterator` use global batch size.\n\n    `make_input_fn_iterator` assumes per-replica batching.\n\n    Returns:\n      Boolean.\n    '
        return True

    def _in_multi_worker_mode(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether this strategy indicates working in multi-worker settings.'
        return False

    def _get_local_replica_id(self, replica_id_in_sync_group):
        if False:
            for i in range(10):
                print('nop')
        return replica_id_in_sync_group

    def _get_replica_id_in_sync_group(self, replica_id):
        if False:
            for i in range(10):
                print('nop')
        return replica_id