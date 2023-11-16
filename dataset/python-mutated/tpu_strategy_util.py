"""TPU specific APIs to be used in conjunction with TPU Strategy."""
import gc
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
_INITIALIZED_TPU_SYSTEMS = {}
_LOCAL_MASTERS = ('', 'local')
_tpu_worker_address = monitoring.StringGauge('/tensorflow/tpu/worker_address', 'The worker address that the coordinator/client connects to.', 'address')

def initialize_tpu_system_impl(cluster_resolver, tpu_cluster_resolver_cls):
    if False:
        return 10
    'Implementation for tpu.experimental.initialize_tpu_system.\n\n  Kept separate to avoid tpu_oss code duplication.\n\n  Initialize the TPU devices.\n\n  Args:\n    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n        which provides information about the TPU cluster.\n    tpu_cluster_resolver_cls: a reference to\n        tf.distribute.cluster_resolver.TPUClusterResolver so that an instance\n        of it can be initialized if cluster_resolver is None.\n  Returns:\n    The tf.tpu.Topology object for the topology of the TPU cluster. If called\n    inside tf.function, it returns the serialized topology object instead.\n\n  Raises:\n    RuntimeError: If running inside a tf.function.\n    NotFoundError: If no TPU devices found in eager mode.\n    TypeError: If tpu_cluster_resolver_cls is\n        not tf.distribute.cluster_resolver.TPUClusterResolver.\n  '
    if tpu_cluster_resolver_cls is None or not issubclass(tpu_cluster_resolver_cls, cluster_resolver_lib.ClusterResolver) or (not hasattr(tpu_cluster_resolver_cls, 'tpu_hardware_feature')):
        raise TypeError('tpu_cluster_resolver_cls is not tf.distribute.cluster_resolver.TPUClusterResolver.')
    logging.info('Deallocate tpu buffers before initializing tpu system.')
    context.context()._clear_caches()
    context.context().clear_kernel_cache()
    gc.collect()
    job = None
    if cluster_resolver is None:
        if context.executing_eagerly():
            curr_device = device.DeviceSpec.from_string(context.context().device_name)
            if curr_device.job is not None:
                job = '{}/replica:0/task:0'.format(curr_device.job)
        cluster_resolver = tpu_cluster_resolver_cls('')
    assert isinstance(cluster_resolver, tpu_cluster_resolver_cls)
    tpu_name = compat.as_text(cluster_resolver._tpu)
    if tpu_name in _INITIALIZED_TPU_SYSTEMS:
        logging.warning('TPU system %s has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.', tpu_name)
    logging.info('Initializing the TPU system: %s', tpu_name)
    if tpu_name not in _LOCAL_MASTERS:
        job = '{}/replica:0/task:0'.format(cluster_resolver.get_job_name())
    if context.executing_eagerly():

        @def_function.function(autograph=False)
        def _tpu_init_fn():
            if False:
                while True:
                    i = 10
            return tpu.initialize_system(job=job, compilation_failure_closes_chips=False, tpu_cancellation_closes_chips=False)
        run_eagerly = def_function.functions_run_eagerly()
        if run_eagerly:
            logging.warning('It looks like tf.function behavior was disabled, perhaps using tf.config.run_functions_eagerly. tf.tpu.experimental.initialize_tpu_system requires tf.function to work. This primitive will override the disable.')
            def_function.run_functions_eagerly(False)
        try:
            with ops.device(tpu._tpu_system_device_name(job)):
                output = _tpu_init_fn()
            context.async_wait()
        except errors.InvalidArgumentError as e:
            raise errors.NotFoundError(None, None, 'TPUs not found in the cluster. Failed in initialization: ' + str(e))
        finally:
            if run_eagerly is not None:
                def_function.run_functions_eagerly(run_eagerly)
        context.context()._initialize_logical_devices()
        serialized_topology = output.numpy()
    elif not ops.executing_eagerly_outside_functions():
        master = cluster_resolver.master()
        cluster_spec = cluster_resolver.cluster_spec()
        session_config = config_pb2.ConfigProto(allow_soft_placement=True)
        if cluster_spec:
            session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        with ops.Graph().as_default():
            with session_lib.Session(config=session_config, target=master) as sess:
                serialized_topology = sess.run(tpu.initialize_system())
    else:
        with ops.device(tpu._tpu_system_device_name(job)):
            serialized_topology = tpu.initialize_system(job=job, compilation_failure_closes_chips=False)
            return serialized_topology
    logging.info('Finished initializing TPU system.')
    tpu_topology = topology.Topology(serialized=serialized_topology)
    cluster_resolver.set_tpu_topology(serialized_topology)
    _INITIALIZED_TPU_SYSTEMS[tpu_name] = tpu_topology
    _tpu_worker_address.get_cell('address').set(cluster_resolver.get_master())
    return tpu_topology

def get_initialized_tpu_systems():
    if False:
        i = 10
        return i + 15
    'Returns all currently initialized tpu systems.\n\n  Returns:\n     A dictionary, with tpu name as the key and the tpu topology as the value.\n  '
    return _INITIALIZED_TPU_SYSTEMS.copy()

def shutdown_tpu_system_impl(cluster_resolver, tpu_cluster_resolver_cls):
    if False:
        print('Hello World!')
    'Implementation for tpu.experimental.shutdown_tpu_system.\n\n  Kept separate to avoid tpu_oss code duplication.\n\n  Shuts down the TPU devices.\n\n  This will clear all caches, even those that are maintained through sequential\n  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation\n  cache.\n\n  Args:\n    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n        which provides information about the TPU cluster.\n    tpu_cluster_resolver_cls: a reference to\n        tf.distribute.cluster_resolver.TPUClusterResolver so that an instance\n        of it can be initialized if cluster_resolver is None.\n\n  Raises:\n    RuntimeError: If no TPU devices found for eager execution or if run in a\n        tf.function.\n    TypeError: If tpu_cluster_resolver_cls is\n        not tf.distribute.cluster_resolver.TPUClusterResolver.\n  '
    if tpu_cluster_resolver_cls is None or not issubclass(tpu_cluster_resolver_cls, cluster_resolver_lib.ClusterResolver) or (not hasattr(tpu_cluster_resolver_cls, 'tpu_hardware_feature')):
        raise TypeError('tpu_cluster_resolver_cls is not tf.distribute.cluster_resolver.TPUClusterResolver.')
    job = None
    if cluster_resolver is None:
        if context.executing_eagerly():
            curr_device = device.DeviceSpec.from_string(context.context().device_name)
            if curr_device.job is not None:
                job = '{}/replica:0/task:0'.format(curr_device.job)
        cluster_resolver = tpu_cluster_resolver_cls('')
    assert isinstance(cluster_resolver, tpu_cluster_resolver_cls)
    tpu_name = compat.as_text(cluster_resolver._tpu)
    if tpu_name not in _INITIALIZED_TPU_SYSTEMS:
        logging.warning('You are shutting down a TPU system %s that has not been initialized.' % tpu_name)
    logging.info('Shutting down the TPU system: %s', tpu_name)
    if context.executing_eagerly():
        if tpu_name not in _LOCAL_MASTERS:
            job = '{}/replica:0/task:0'.format(cluster_resolver.get_job_name())

        @def_function.function(autograph=False)
        def _tpu_shutdown_fn():
            if False:
                return 10
            tpu.shutdown_system(job=job)
        run_eagerly = def_function.functions_run_eagerly()
        if run_eagerly:
            logging.warning('It looks like tf.function behavior was disabled, perhaps using tf.config.run_functions_eagerly. tf.tpu.experimental.shutdown_tpu_system requires tf.function to work. This primitive will override the disable.')
            def_function.run_functions_eagerly(False)
        try:
            with ops.device(tpu._tpu_system_device_name(job)):
                _tpu_shutdown_fn()
        finally:
            if run_eagerly is not None:
                def_function.run_functions_eagerly(run_eagerly)
        logging.info('Clearing out eager caches')
        context.context()._clear_caches()
        context.context().clear_kernel_cache()
    elif not ops.executing_eagerly_outside_functions():
        master = cluster_resolver.master()
        cluster_spec = cluster_resolver.cluster_spec()
        session_config = config_pb2.ConfigProto(allow_soft_placement=True)
        if cluster_spec:
            session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        with ops.Graph().as_default():
            with session_lib.Session(config=session_config, target=master) as sess:
                sess.run(tpu.shutdown_system())
    else:
        raise RuntimeError('initialize_tpu_system is not supported within tf.functions.  You should call initialize_tpu_system outside of your tf.function. ')
    logging.info('Finished shutting down TPU system.')
    if tpu_name in _INITIALIZED_TPU_SYSTEMS:
        del _INITIALIZED_TPU_SYSTEMS[tpu_name]