"""Implementation of Cluster Resolvers for Cloud TPUs."""
import collections
import re
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.eager import remote
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
try:
    from cloud_tpu_client import client
except ImportError:
    logging.debug('Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client.')
    from tensorflow.python.tpu.client import client

def is_running_in_gce():
    if False:
        i = 10
        return i + 15
    return True

class _LocalCloudTpuClient(object):
    """Dummy local Cloud TPU client."""

    def api_available(self):
        if False:
            i = 10
            return i + 15
        return False
_TPU_DEVICE_REGEX = re.compile('.*task:(?P<host_id>\\d+)/.*device:TPU:(?P<core_id>\\d+)$')
_TPU_CONN_RETRIES = 120
DeviceDetails = collections.namedtuple('DeviceDetails', ['device_map', 'total_cores'])

def initialize_tpu_system(cluster_resolver=None):
    if False:
        i = 10
        return i + 15
    'Initialize the TPU devices.\n\n  Args:\n    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n        which provides information about the TPU cluster.\n  Returns:\n    The tf.tpu.Topology object for the topology of the TPU cluster. If called\n    inside tf.function, it returns the serialized topology object instead.\n\n  Raises:\n    RuntimeError: If running inside a tf.function.\n    NotFoundError: If no TPU devices found in eager mode.\n  '
    return tpu_strategy_util.initialize_tpu_system_impl(cluster_resolver, TPUClusterResolver)

def shutdown_tpu_system(cluster_resolver=None):
    if False:
        i = 10
        return i + 15
    'Shuts down the TPU devices.\n\n  This will clear all caches, even those that are maintained through sequential\n  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation\n  cache.\n\n  Args:\n    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,\n        which provides information about the TPU cluster.\n\n  Raises:\n    RuntimeError: If no TPU devices found for eager execution or if run in a\n        tf.function.\n  '
    tpu_strategy_util.shutdown_tpu_system_impl(cluster_resolver, TPUClusterResolver)

class TPUClusterResolver(cluster_resolver_lib.ClusterResolver):
    """Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service.

  TPUClusterResolver supports the following distinct environments:
  Google Compute Engine
  Google Kubernetes Engine
  Google internal

  It can be passed into `tf.distribute.TPUStrategy` to support TF2 training on
  Cloud TPUs.
  """

    @staticmethod
    def connect(tpu=None, zone=None, project=None):
        if False:
            while True:
                i = 10
        "Initializes TPU and returns a TPUClusterResolver.\n\n    This API will connect to remote TPU cluster and initialize the TPU\n    hardwares. Example usage:\n\n    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(\n    ...     tpu='')\n\n    It can be viewed as convenient wrapper of the following code:\n\n    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    >>> tf.config.experimental_connect_to_cluster(resolver)\n    >>> tf.tpu.experimental.initialize_tpu_system(resolver)\n\n    Args:\n      tpu: A string corresponding to the TPU to use. It can be the TPU name or\n        TPU worker gRPC address. If not set, it will try automatically resolve\n        the TPU address on Cloud TPUs.\n      zone: Zone where the TPUs are located. If omitted or empty, we will assume\n        that the zone of the TPU is the same as the zone of the GCE VM, which we\n        will try to discover from the GCE metadata service.\n      project: Name of the GCP project containing Cloud TPUs. If omitted or\n        empty, we will try to discover the project name of the GCE VM from the\n        GCE metadata service.\n\n    Returns:\n      An instance of TPUClusterResolver object.\n\n    Raises:\n      NotFoundError: If no TPU devices found in eager mode.\n    "
        resolver = TPUClusterResolver(tpu, zone, project)
        remote.connect_to_cluster(resolver)
        tpu_strategy_util.initialize_tpu_system_impl(resolver)
        return resolver

    @staticmethod
    def _get_device_dict_and_cores(devices):
        if False:
            return 10
        'Returns a dict of hosts to cores and total cores given devices names.\n\n    Returns a namedtuple with two attributes:\n      device_map: A map of host_ids to a list of core_ids.\n      total_cores: The total number of cores within the TPU system.\n\n    Args:\n      devices: A list of devices returned by session.list_devices()\n    '
        device_map = collections.defaultdict(list)
        num_cores = 0
        for device in devices:
            match = _TPU_DEVICE_REGEX.match(device.name)
            if match:
                host_id = match.group('host_id')
                core_id = match.group('core_id')
                device_map[host_id].append(core_id)
                num_cores += 1
        return DeviceDetails(device_map, num_cores)

    @staticmethod
    def _verify_and_return_same_core_count(device_dict):
        if False:
            print('Hello World!')
        'Verifies that every device in device_dict has the same # of cores.'
        num_cores_per_host_set = {len(core_ids) for core_ids in device_dict.values()}
        if len(num_cores_per_host_set) != 1:
            raise RuntimeError('TPU cores on each device is not the same. This should never happen. Devices: {}'.format(device_dict))
        return num_cores_per_host_set.pop()

    def __init__(self, tpu=None, zone=None, project=None, job_name='worker', coordinator_name=None, coordinator_address=None, credentials='default', service=None, discovery_url=None):
        if False:
            print('Hello World!')
        'Creates a new TPUClusterResolver object.\n\n    The ClusterResolver will then use the parameters to query the Cloud TPU APIs\n    for the IP addresses and ports of each Cloud TPU listed.\n\n    Args:\n      tpu: A string corresponding to the TPU to use. It can be the TPU name or\n        TPU worker gRPC address. If not set, it will try automatically resolve\n        the TPU address on Cloud TPUs. If set to "local", it will assume that\n        the TPU is directly connected to the VM instead of over the network.\n      zone: Zone where the TPUs are located. If omitted or empty, we will assume\n        that the zone of the TPU is the same as the zone of the GCE VM, which we\n        will try to discover from the GCE metadata service.\n      project: Name of the GCP project containing Cloud TPUs. If omitted or\n        empty, we will try to discover the project name of the GCE VM from the\n        GCE metadata service.\n      job_name: Name of the TensorFlow job the TPUs belong to.\n      coordinator_name: The name to use for the coordinator. Set to None if the\n        coordinator should not be included in the computed ClusterSpec.\n      coordinator_address: The address of the coordinator (typically an ip:port\n        pair). If set to None, a TF server will be started. If coordinator_name\n        is None, a TF server will not be started even if coordinator_address is\n        None.\n      credentials: GCE Credentials. If None, then we use default credentials\n        from the oauth2client\n      service: The GCE API object returned by the googleapiclient.discovery\n        function. If you specify a custom service object, then the credentials\n        parameter will be ignored.\n      discovery_url: A URL template that points to the location of the discovery\n        service. It should have two parameters {api} and {apiVersion} that when\n        filled in produce an absolute URL to the discovery document for that\n        service. The environment variable \'TPU_API_DISCOVERY_URL\' will override\n        this.\n\n    Raises:\n      ImportError: If the googleapiclient is not installed.\n      ValueError: If no TPUs are specified.\n      RuntimeError: If an empty TPU name is specified and this is running in a\n        Google Cloud environment.\n    '
        if tpu != 'local':
            self._cloud_tpu_client = client.Client(tpu=tpu, zone=zone, project=project, credentials=credentials, service=service, discovery_url=discovery_url)
            self._tpu = self._cloud_tpu_client.name()
        else:
            self._cloud_tpu_client = _LocalCloudTpuClient()
            self._tpu = 'local'
        self.task_type = job_name
        self.task_id = 0
        self._coordinator_name = coordinator_name
        if coordinator_name and (not coordinator_address):
            self._start_local_server()
        else:
            self._coordinator_address = coordinator_address
        self._tpu_topology = None

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self._cloud_tpu_client.enter()

    def __exit__(self, type, value, traceback):
        if False:
            print('Hello World!')
        self._cloud_tpu_client.exit(type, value, traceback)

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            for i in range(10):
                print('nop')
        "Get the Master string to be used for the session.\n\n    In the normal case, this returns the grpc path (grpc://1.2.3.4:8470) of\n    first instance in the ClusterSpec returned by the cluster_spec function.\n\n    If a non-TPU name is used when constructing a TPUClusterResolver, that will\n    be returned instead (e.g. If the tpus argument's value when constructing\n    this TPUClusterResolver was 'grpc://10.240.1.2:8470',\n    'grpc://10.240.1.2:8470' will be returned).\n\n    Args:\n      task_type: (Optional, string) The type of the TensorFlow task of the\n        master.\n      task_id: (Optional, integer) The index of the TensorFlow task of the\n        master.\n      rpc_layer: (Optional, string) The RPC protocol TensorFlow should use to\n        communicate with TPUs.\n\n    Returns:\n      string, the connection string to use when creating a session.\n\n    Raises:\n      ValueError: If none of the TPUs specified exists.\n    "
        if self._tpu != 'local':
            cluster_spec = self.cluster_spec()
            if task_type is not None and task_id is not None:
                master = cluster_spec.task_address(task_type, task_id)
            elif self.task_type is not None and self.task_id is not None:
                master = cluster_spec.task_address(self.task_type, self.task_id)
            else:
                job_tasks = cluster_spec.job_tasks(self.task_type)
                if not job_tasks:
                    raise ValueError('No TPUs with the specified names exist.')
                master = job_tasks[0]
            return cluster_resolver_lib.format_master_url(master, 'grpc')
        else:
            return ''

    def get_master(self):
        if False:
            return 10
        return self.master()

    def get_job_name(self):
        if False:
            return 10
        return self.task_type

    def get_coordination_service_leader(self):
        if False:
            return 10
        'Returns the location for coordination service.\n\n    The coordination service should be located on TPU worker0.\n\n    Returns:\n      A string indicate the location path.\n    '
        return '/job:' + self.get_job_name() + '/task:0'

    def get_tpu_system_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the metadata of the TPU system.\n\n    Users can call this method to get some facts of the TPU system, like\n    total number of cores, number of TPU workers and the devices. E.g.\n    ```python\n\n    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')\n    tpu_system_metadata = resolver.get_tpu_system_metadata()\n    num_hosts = tpu_system_metadata.num_hosts\n    ```\n\n    Returns:\n      A `tf.tpu.experimental.TPUSystemMetadata` object.\n    "
        cluster_spec = self.cluster_spec()
        cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
        tpu_system_metadata = tpu_system_metadata_lib._query_tpu_system_metadata(self.master(), cluster_def=cluster_def, query_topology=False)
        return tpu_system_metadata

    def cluster_spec(self):
        if False:
            return 10
        'Returns a ClusterSpec object based on the latest TPU information.\n\n    We retrieve the information from the GCE APIs every time this method is\n    called.\n\n    Returns:\n      A ClusterSpec containing host information returned from Cloud TPUs,\n      or None.\n\n    Raises:\n      RuntimeError: If the provided TPU is not healthy.\n    '
        if self._tpu != 'local':
            network_endpoints = self._cloud_tpu_client.network_endpoints()
            worker_list = ['%s:%s' % (endpoint['ipAddress'], endpoint['port']) for endpoint in network_endpoints]
            cluster_spec = {self.task_type: worker_list}
            if self._coordinator_address:
                cluster_spec[self._coordinator_name] = [self._coordinator_address]
            return server_lib.ClusterSpec(cluster_spec)
        else:
            return server_lib.ClusterSpec({})

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        if False:
            return 10
        'Returns the number of TPU cores per worker.\n\n    Connects to the master and list all the devices present in the master,\n    and counts them up. Also verifies that the device counts per host in the\n    cluster is the same before returning the number of TPU cores per host.\n\n    Args:\n      task_type: Unused.\n      task_id: Unused.\n      config_proto: Used to create a connection to a TPU master in order to\n        retrieve the system metadata.\n\n    Raises:\n      RuntimeError: If we cannot talk to a TPU worker after retrying or if the\n        number of TPU devices per host is different.\n    '
        if self._tpu == 'local':
            return {'TPU': len([d for d in framework_config.list_logical_devices() if d.device_type == 'TPU'])}
        retry_count = 1
        while True:
            try:
                device_details = TPUClusterResolver._get_device_dict_and_cores(cluster_resolver_lib.get_accelerator_devices(self.master(), config_proto=config_proto))
                break
            except errors.DeadlineExceededError:
                error_message = 'Failed to connect to master. The TPU might not be ready (e.g. still scheduling) or the master address is incorrect: got (%s)' % self.master()
                if retry_count <= _TPU_CONN_RETRIES:
                    logging.warning(error_message)
                    logging.warning('Retrying (%d/%d)...', retry_count, _TPU_CONN_RETRIES)
                    retry_count += 1
                else:
                    raise RuntimeError(error_message)
        if device_details.total_cores:
            return {'TPU': TPUClusterResolver._verify_and_return_same_core_count(device_details.device_map)}
        return {'TPU': 0}

    def set_tpu_topology(self, serialized_tpu_topology):
        if False:
            print('Hello World!')
        'Sets the tpu topology info stored in this resolver.'
        self._tpu_topology = topology_pb2.TopologyProto()
        self._tpu_topology.ParseFromString(serialized_tpu_topology)

    @property
    def tpu_hardware_feature(self):
        if False:
            return 10
        'Returns the tpu topology info stored.'
        if self._tpu_topology is None:
            return self._tpu_topology
        return self._tpu_topology.tpu_hardware_feature

    @property
    def environment(self):
        if False:
            while True:
                i = 10
        'Returns the current environment which TensorFlow is running in.'
        return ''

    def _start_local_server(self):
        if False:
            print('Hello World!')
        address = compat.as_text(self._cloud_tpu_client.get_local_ip())
        self._server = server_lib.Server({'local': ['0.0.0.0:0']}, protocol='grpc', config=None, start=True)
        target = compat.as_bytes(self._server.target)
        splits = target.split(compat.as_bytes(':'))
        assert len(splits) == 3, self._server.target
        assert splits[0] == compat.as_bytes('grpc'), self._server.target
        self._coordinator_port = compat.as_text(splits[2])
        self._coordinator_address = '%s:%s' % (address, compat.as_text(self._coordinator_port))

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        return self