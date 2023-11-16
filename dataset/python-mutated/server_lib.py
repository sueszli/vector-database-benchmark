"""A Python interface for creating TensorFlow servers."""
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

def _make_server_def(server_or_cluster_def, job_name, task_index, protocol, config):
    if False:
        print('Hello World!')
    'Creates a `tf.train.ServerDef` protocol buffer.\n\n  Args:\n    server_or_cluster_def: A `tf.train.ServerDef` or `tf.train.ClusterDef`\n      protocol buffer, or a `tf.train.ClusterSpec` object, describing the server\n      to be defined and/or the cluster of which it is a member.\n    job_name: (Optional.) Specifies the name of the job of which the server is a\n      member. Defaults to the value in `server_or_cluster_def`, if specified.\n    task_index: (Optional.) Specifies the task index of the server in its job.\n      Defaults to the value in `server_or_cluster_def`, if specified. Otherwise\n      defaults to 0 if the server\'s job has only one task.\n    protocol: (Optional.) Specifies the protocol to be used by the server.\n      Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value in\n      `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.\n    config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default\n      configuration options for all sessions that run on this server.\n\n  Returns:\n    A `tf.train.ServerDef`.\n\n  Raises:\n    TypeError: If the arguments do not have the appropriate type.\n    ValueError: If an argument is not specified and cannot be inferred.\n  '
    server_def = tensorflow_server_pb2.ServerDef()
    if isinstance(server_or_cluster_def, tensorflow_server_pb2.ServerDef):
        server_def.MergeFrom(server_or_cluster_def)
        if job_name is not None:
            server_def.job_name = job_name
        if task_index is not None:
            server_def.task_index = task_index
        if protocol is not None:
            server_def.protocol = protocol
        if config is not None:
            server_def.default_session_config.MergeFrom(config)
    else:
        try:
            cluster_spec = ClusterSpec(server_or_cluster_def)
        except TypeError:
            raise TypeError('Could not convert `server_or_cluster_def` to a `tf.train.ServerDef` or `tf.train.ClusterSpec`.')
        if job_name is None:
            if len(cluster_spec.jobs) == 1:
                job_name = cluster_spec.jobs[0]
            else:
                raise ValueError('Must specify an explicit `job_name`.')
        if task_index is None:
            task_indices = cluster_spec.task_indices(job_name)
            if len(task_indices) == 1:
                task_index = task_indices[0]
            else:
                raise ValueError('Must specify an explicit `task_index`.')
        if protocol is None:
            protocol = 'grpc'
        server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_spec.as_cluster_def(), job_name=job_name, task_index=task_index, protocol=protocol)
        if config is not None:
            server_def.default_session_config.MergeFrom(config)
    return server_def

@tf_export('distribute.Server', v1=['distribute.Server', 'train.Server'])
@deprecation.deprecated_endpoints('train.Server')
class Server:
    """An in-process TensorFlow server, for use in distributed training.

  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  """

    def __init__(self, server_or_cluster_def, job_name=None, task_index=None, protocol=None, config=None, start=True):
        if False:
            while True:
                i = 10
        'Creates a new server with the given definition.\n\n    The `job_name`, `task_index`, and `protocol` arguments are optional, and\n    override any information provided in `server_or_cluster_def`.\n\n    Args:\n      server_or_cluster_def: A `tf.train.ServerDef` or `tf.train.ClusterDef`\n        protocol buffer, or a `tf.train.ClusterSpec` object, describing the\n        server to be created and/or the cluster of which it is a member.\n      job_name: (Optional.) Specifies the name of the job of which the server is\n        a member. Defaults to the value in `server_or_cluster_def`, if\n        specified.\n      task_index: (Optional.) Specifies the task index of the server in its job.\n        Defaults to the value in `server_or_cluster_def`, if specified.\n        Otherwise defaults to 0 if the server\'s job has only one task.\n      protocol: (Optional.) Specifies the protocol to be used by the server.\n        Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value\n        in `server_or_cluster_def`, if specified. Otherwise defaults to\n        `"grpc"`.\n      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default\n        configuration options for all sessions that run on this server.\n      start: (Optional.) Boolean, indicating whether to start the server after\n        creating it. Defaults to `True`.\n\n    Raises:\n      tf.errors.OpError: Or one of its subclasses if an error occurs while\n        creating the TensorFlow server.\n    '
        self._server_def = _make_server_def(server_or_cluster_def, job_name, task_index, protocol, config)
        self._server = c_api.TF_NewServer(self._server_def.SerializeToString())
        if start:
            self.start()

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if errors is not None:
            exception = errors.UnimplementedError
        else:
            exception = Exception
        try:
            c_api.TF_ServerStop(self._server)
        except AttributeError:
            pass
        except exception:
            pass
        self._server = None

    def start(self):
        if False:
            return 10
        'Starts this server.\n\n    Raises:\n      tf.errors.OpError: Or one of its subclasses if an error occurs while\n        starting the TensorFlow server.\n    '
        c_api.TF_ServerStart(self._server)

    def join(self):
        if False:
            i = 10
            return i + 15
        'Blocks until the server has shut down.\n\n    This method currently blocks forever.\n\n    Raises:\n      tf.errors.OpError: Or one of its subclasses if an error occurs while\n        joining the TensorFlow server.\n    '
        c_api.TF_ServerJoin(self._server)

    @property
    def server_def(self):
        if False:
            while True:
                i = 10
        'Returns the `tf.train.ServerDef` for this server.\n\n    Returns:\n      A `tf.train.ServerDef` protocol buffer that describes the configuration\n      of this server.\n    '
        return self._server_def

    @property
    def target(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the target for a `tf.compat.v1.Session` to connect to this server.\n\n    To create a\n    `tf.compat.v1.Session` that\n    connects to this server, use the following snippet:\n\n    ```python\n    server = tf.distribute.Server(...)\n    with tf.compat.v1.Session(server.target):\n      # ...\n    ```\n\n    Returns:\n      A string containing a session target for this server.\n    '
        return c_api.TF_ServerTarget(self._server)

    @staticmethod
    def create_local_server(config=None, start=True):
        if False:
            for i in range(10):
                print('nop')
        'Creates a new single-process cluster running on the local host.\n\n    This method is a convenience wrapper for creating a\n    `tf.distribute.Server` with a `tf.train.ServerDef` that specifies a\n    single-process cluster containing a single task in a job called\n    `"local"`.\n\n    Args:\n      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default\n        configuration options for all sessions that run on this server.\n      start: (Optional.) Boolean, indicating whether to start the server after\n        creating it. Defaults to `True`.\n\n    Returns:\n      A local `tf.distribute.Server`.\n    '
        return Server({'localhost': ['localhost:0']}, protocol='grpc', config=config, start=start)

@tf_export('train.ClusterSpec')
class ClusterSpec:
    """Represents a cluster as a set of "tasks", organized into "jobs".

  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  `tf.distribute.Server` is constructed in a particular cluster.

  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).

  ```python
  cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                             "worker1.example.com:2222",
                                             "worker2.example.com:2222"],
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```

  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:

  ```python
  cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```
  """

    def __init__(self, cluster):
        if False:
            while True:
                i = 10
        'Creates a `ClusterSpec`.\n\n    Args:\n      cluster: A dictionary mapping one or more job names to (i) a list of\n        network addresses, or (ii) a dictionary mapping integer task indices to\n        network addresses; or a `tf.train.ClusterDef` protocol buffer.\n\n    Raises:\n      TypeError: If `cluster` is not a dictionary mapping strings to lists\n        of strings, and not a `tf.train.ClusterDef` protobuf.\n    '
        if isinstance(cluster, dict):
            self._cluster_spec = {}
            for (job_name, tasks) in cluster.items():
                if isinstance(tasks, (list, tuple)):
                    job_tasks = {i: task for (i, task) in enumerate(tasks)}
                elif isinstance(tasks, dict):
                    job_tasks = {int(i): task for (i, task) in tasks.items()}
                else:
                    raise TypeError('The tasks for job %r must be a list or a dictionary from integers to strings.' % job_name)
                self._cluster_spec[job_name] = job_tasks
            self._make_cluster_def()
        elif isinstance(cluster, cluster_pb2.ClusterDef):
            self._cluster_def = cluster
            self._cluster_spec = {}
            for job_def in self._cluster_def.job:
                self._cluster_spec[job_def.name] = {i: t for (i, t) in job_def.tasks.items()}
        elif isinstance(cluster, ClusterSpec):
            self._cluster_def = cluster_pb2.ClusterDef()
            self._cluster_def.MergeFrom(cluster.as_cluster_def())
            self._cluster_spec = {}
            for job_def in self._cluster_def.job:
                self._cluster_spec[job_def.name] = {i: t for (i, t) in job_def.tasks.items()}
        else:
            raise TypeError('`cluster` must be a dictionary mapping one or more job names to lists of network addresses, or a `ClusterDef` protocol buffer')

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self._cluster_spec)
    __nonzero__ = __bool__

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self._cluster_spec == other

    def __ne__(self, other):
        if False:
            return 10
        return self._cluster_spec != other

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        key_values = self.as_dict()
        string_items = [repr(k) + ': ' + repr(key_values[k]) for k in sorted(key_values)]
        return 'ClusterSpec({' + ', '.join(string_items) + '})'

    def as_dict(self):
        if False:
            return 10
        'Returns a dictionary from job names to their tasks.\n\n    For each job, if the task index space is dense, the corresponding\n    value will be a list of network addresses; otherwise it will be a\n    dictionary mapping (sparse) task indices to the corresponding\n    addresses.\n\n    Returns:\n      A dictionary mapping job names to lists or dictionaries\n      describing the tasks in those jobs.\n    '
        ret = {}
        for job in self.jobs:
            task_indices = self.task_indices(job)
            if len(task_indices) == 0:
                ret[job] = {}
                continue
            if max(task_indices) + 1 == len(task_indices):
                ret[job] = self.job_tasks(job)
            else:
                ret[job] = {i: self.task_address(job, i) for i in task_indices}
        return ret

    def as_cluster_def(self):
        if False:
            print('Hello World!')
        'Returns a `tf.train.ClusterDef` protocol buffer based on this cluster.'
        return self._cluster_def

    @property
    def jobs(self):
        if False:
            return 10
        'Returns a list of job names in this cluster.\n\n    Returns:\n      A list of strings, corresponding to the names of jobs in this cluster.\n    '
        return list(self._cluster_spec.keys())

    def num_tasks(self, job_name):
        if False:
            i = 10
            return i + 15
        'Returns the number of tasks defined in the given job.\n\n    Args:\n      job_name: The string name of a job in this cluster.\n\n    Returns:\n      The number of tasks defined in the given job.\n\n    Raises:\n      ValueError: If `job_name` does not name a job in this cluster.\n    '
        try:
            job = self._cluster_spec[job_name]
        except KeyError:
            raise ValueError('No such job in cluster: %r' % job_name)
        return len(job)

    def task_indices(self, job_name):
        if False:
            i = 10
            return i + 15
        'Returns a list of valid task indices in the given job.\n\n    Args:\n      job_name: The string name of a job in this cluster.\n\n    Returns:\n      A list of valid task indices in the given job.\n\n    Raises:\n      ValueError: If `job_name` does not name a job in this cluster,\n      or no task with index `task_index` is defined in that job.\n    '
        try:
            job = self._cluster_spec[job_name]
        except KeyError:
            raise ValueError('No such job in cluster: %r' % job_name)
        return list(sorted(job.keys()))

    def task_address(self, job_name, task_index):
        if False:
            i = 10
            return i + 15
        'Returns the address of the given task in the given job.\n\n    Args:\n      job_name: The string name of a job in this cluster.\n      task_index: A non-negative integer.\n\n    Returns:\n      The address of the given task in the given job.\n\n    Raises:\n      ValueError: If `job_name` does not name a job in this cluster,\n      or no task with index `task_index` is defined in that job.\n    '
        try:
            job = self._cluster_spec[job_name]
        except KeyError:
            raise ValueError('No such job in cluster: %r' % job_name)
        try:
            return job[task_index]
        except KeyError:
            raise ValueError('No task with index %r in job %r' % (task_index, job_name))

    def job_tasks(self, job_name):
        if False:
            i = 10
            return i + 15
        'Returns a mapping from task ID to address in the given job.\n\n    NOTE: For backwards compatibility, this method returns a list. If\n    the given job was defined with a sparse set of task indices, the\n    length of this list may not reflect the number of tasks defined in\n    this job. Use the `tf.train.ClusterSpec.num_tasks` method\n    to find the number of tasks defined in a particular job.\n\n    Args:\n      job_name: The string name of a job in this cluster.\n\n    Returns:\n      A list of task addresses, where the index in the list\n      corresponds to the task index of each task. The list may contain\n      `None` if the job was defined with a sparse set of task indices.\n\n    Raises:\n      ValueError: If `job_name` does not name a job in this cluster.\n    '
        try:
            job = self._cluster_spec[job_name]
        except KeyError:
            raise ValueError('No such job in cluster: %r' % job_name)
        ret = [None for _ in range(max(job.keys()) + 1)]
        for (i, task) in job.items():
            ret[i] = task
        return ret

    def _make_cluster_def(self):
        if False:
            while True:
                i = 10
        'Creates a `tf.train.ClusterDef` based on the given `cluster_spec`.\n\n    Raises:\n      TypeError: If `cluster_spec` is not a dictionary mapping strings to lists\n        of strings.\n    '
        self._cluster_def = cluster_pb2.ClusterDef()
        for (job_name, tasks) in sorted(self._cluster_spec.items()):
            try:
                job_name = compat.as_bytes(job_name)
            except TypeError:
                raise TypeError('Job name %r must be bytes or unicode' % job_name)
            job_def = self._cluster_def.job.add()
            job_def.name = job_name
            for (i, task_address) in sorted(tasks.items()):
                try:
                    task_address = compat.as_bytes(task_address)
                except TypeError:
                    raise TypeError('Task address %r must be bytes or unicode' % task_address)
                job_def.tasks[i] = task_address

@tf_export('config.experimental.ClusterDeviceFilters')
class ClusterDeviceFilters:
    """Represent a collection of device filters for the remote workers in cluster.

  NOTE: this is an experimental API and subject to changes.

  Set device filters for selective jobs and tasks. For each remote worker, the
  device filters are a list of strings. When any filters are present, the remote
  worker will ignore all devices which do not match any of its filters. Each
  filter can be partially specified, e.g. "/job:ps", "/job:worker/replica:3",
  etc. Note that a device is always visible to the worker it is located on.

  For example, to set the device filters for a parameter server cluster:

  ```python
  cdf = tf.config.experimental.ClusterDeviceFilters()
  for i in range(num_workers):
    cdf.set_device_filters('worker', i, ['/job:ps'])
  for i in range(num_ps):
    cdf.set_device_filters('ps', i, ['/job:worker'])

  tf.config.experimental_connect_to_cluster(cluster_def,
                                            cluster_device_filters=cdf)
  ```

  The device filters can be partically specified. For remote tasks that do not
  have device filters specified, all devices will be visible to them.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._device_filters = {}
        self._cluster_device_filters = None

    def set_device_filters(self, job_name, task_index, device_filters):
        if False:
            i = 10
            return i + 15
        'Set the device filters for given job name and task id.'
        assert all((isinstance(df, str) for df in device_filters))
        self._device_filters.setdefault(job_name, {})
        self._device_filters[job_name][task_index] = [df for df in device_filters]
        self._cluster_device_filters = None

    def _as_cluster_device_filters(self):
        if False:
            i = 10
            return i + 15
        'Returns a serialized protobuf of cluster device filters.'
        if self._cluster_device_filters:
            return self._cluster_device_filters
        self._make_cluster_device_filters()
        return self._cluster_device_filters

    def _make_cluster_device_filters(self):
        if False:
            while True:
                i = 10
        'Creates `ClusterDeviceFilters` proto based on the `_device_filters`.\n\n    Raises:\n      TypeError: If `_device_filters` is not a dictionary mapping strings to\n      a map of task indices and device filters.\n    '
        self._cluster_device_filters = device_filters_pb2.ClusterDeviceFilters()
        for (job_name, tasks) in sorted(self._device_filters.items()):
            try:
                job_name = compat.as_bytes(job_name)
            except TypeError:
                raise TypeError('Job name %r must be bytes or unicode' % job_name)
            jdf = self._cluster_device_filters.jobs.add()
            jdf.name = job_name
            for (i, task_device_filters) in sorted(tasks.items()):
                for tdf in task_device_filters:
                    try:
                        tdf = compat.as_bytes(tdf)
                    except TypeError:
                        raise TypeError('Device filter %r must be bytes or unicode' % tdf)
                    jdf.tasks[i].device_filters.append(tdf)