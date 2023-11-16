"""TPU system metadata and associated tooling."""
import collections
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.util.tf_export import tf_export
_PINGING_MASTER_TIMEOUT_IN_MS = 5 * 60 * 1000
_RETRY_TIMES = 12 * 24
_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS = 300 * 1000
_DEFAULT_JOB_NAME = 'tpu_worker'
_DEFAULT_COORDINATOR_JOB_NAME = 'coordinator'
_LOCAL_MASTERS = ('', 'local')

@tf_export('tpu.experimental.TPUSystemMetadata')
class TPUSystemMetadata(collections.namedtuple('TPUSystemMetadata', ['num_cores', 'num_hosts', 'num_of_cores_per_host', 'topology', 'devices'])):
    """Describes some metadata about the TPU system.

  Attributes:
    num_cores: interger. Total number of TPU cores in the TPU system.
    num_hosts: interger. Total number of hosts (TPU workers) in the TPU system.
    num_of_cores_per_host: interger. Number of TPU cores per host (TPU worker).
    topology: an instance of `tf.tpu.experimental.Topology`, which describes the
      physical topology of TPU system.
    devices: a tuple of strings, which describes all the TPU devices in the
      system.
  """

    def __new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology, devices):
        if False:
            i = 10
            return i + 15
        return super(TPUSystemMetadata, cls).__new__(cls, num_cores, num_hosts, num_of_cores_per_host, topology, devices)

def _query_tpu_system_metadata(master_address, cluster_def=None, query_topology=False):
    if False:
        i = 10
        return i + 15
    'Automatically detects the TPU system metadata in the system.'
    tpu_core_count = 0
    devices = []
    device_dict = collections.defaultdict(list)
    if context.executing_eagerly():
        logical_devices = config.list_logical_devices()
        devices = [session_lib._DeviceAttributes(device_util.canonicalize(d.name), d.device_type, 0, 0) for d in logical_devices]
    else:
        retry_count = 1
        while True:
            logging.info('Querying Tensorflow master (%s) for TPU system metadata.', master_address)
            try:
                with ops.Graph().as_default():
                    with session_lib.Session(master_address, config=get_session_config_with_timeout(_PINGING_MASTER_TIMEOUT_IN_MS, cluster_def)) as sess:
                        devices = sess.list_devices()
                        break
            except errors.DeadlineExceededError:
                msg = 'Failed to connect to the Tensorflow master. The TPU worker may not be ready (still scheduling) or the Tensorflow master address is incorrect: got (%s).' % master_address
                if retry_count <= _RETRY_TIMES:
                    logging.warning('%s', msg)
                    logging.warning('Retrying (%d/%d).', retry_count, _RETRY_TIMES)
                    retry_count += 1
                else:
                    raise ValueError(msg)
    for device in devices:
        spec = tf_device.DeviceSpec.from_string(device.name)
        if spec.device_type == 'TPU':
            device_dict[spec.task].append(spec.device_index)
            tpu_core_count += 1
    num_of_cores_per_host = 0
    if tpu_core_count:
        num_cores_per_host_set = set([len(core_ids) for core_ids in device_dict.values()])
        if len(num_cores_per_host_set) != 1:
            raise RuntimeError('TPU cores on each host is not same. This should not happen!. devices: {}'.format(devices))
        num_of_cores_per_host = num_cores_per_host_set.pop()
    topology = None
    if query_topology:
        if not tpu_core_count:
            raise RuntimeError('Cannot find any TPU cores in the system (master address {}). This usually means the master address is incorrect or the TPU worker has some problems. Available devices: {}'.format(master_address, devices))
        topology = _obtain_topology(master_address, cluster_def)

    def _sort_key(device):
        if False:
            return 10
        spec = tf_device.DeviceSpec.from_string(device.name)
        return (spec.job, spec.replica, spec.task, spec.device_type, spec.device_index)
    devices = tuple(sorted(devices, key=_sort_key))
    metadata = TPUSystemMetadata(num_cores=tpu_core_count, num_hosts=len(device_dict), num_of_cores_per_host=num_of_cores_per_host, topology=topology, devices=devices)
    if tpu_core_count:
        logging.info('Found TPU system:')
        logging.info('*** Num TPU Cores: %d', metadata.num_cores)
        logging.info('*** Num TPU Workers: %d', metadata.num_hosts)
        logging.info('*** Num TPU Cores Per Worker: %d', metadata.num_of_cores_per_host)
        for device in metadata.devices:
            logging.info('*** Available Device: %s', device)
    else:
        logging.info('Failed to find TPU: %s', metadata)
    return metadata

def _obtain_topology(master_address, cluster_def):
    if False:
        for i in range(10):
            print('nop')
    'Obtains TPU fabric topology.'
    try:
        logging.info('Initializing TPU system (master: %s) to fetch topology for model parallelism. This might take a while.', master_address)
        with ops.Graph().as_default():
            session_config = get_session_config_with_timeout(_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS, cluster_def)
            with session_lib.Session(master_address, config=session_config) as sess:
                topology = sess.run(tpu.initialize_system())
                return topology
    except errors.DeadlineExceededError:
        raise ValueError('Fail to initialize TPU system with master (%s). Please double check the TPU system is functional.' % master_address)

def get_session_config_with_timeout(timeout_in_secs, cluster_def):
    if False:
        i = 10
        return i + 15
    'Returns a session given a timeout and a cluster configuration.'
    config_proto = config_pb2.ConfigProto(operation_timeout_in_ms=timeout_in_secs, cluster_def=cluster_def)
    return config_proto

def master_job(master, cluster_def):
    if False:
        i = 10
        return i + 15
    'Returns the canonical job name to use to place TPU computations on.\n\n  Args:\n    master: A `string` representing the TensorFlow master to use.\n    cluster_def: A ClusterDef object describing the TPU cluster.\n\n  Returns:\n    A string containing the job name, or None if no job should be specified.\n\n  Raises:\n    ValueError: If the user needs to specify a tpu_job_name, because we are\n      unable to infer the job name automatically, or if the user-specified job\n      names are inappropriate.\n  '
    if master in _LOCAL_MASTERS:
        return None
    if not cluster_def or not cluster_def.job:
        return _DEFAULT_JOB_NAME
    job_names = set((job.name for job in cluster_def.job))
    if _DEFAULT_JOB_NAME in job_names:
        raise ValueError('Currently, tpu_worker is not an allowed job name.')
    if len(job_names) == 1:
        return cluster_def.job[0].name
    if len(job_names) == 2:
        if _DEFAULT_COORDINATOR_JOB_NAME in job_names:
            job_names.remove(_DEFAULT_COORDINATOR_JOB_NAME)
            return job_names.pop()
    raise ValueError('Could not infer TPU job name.')