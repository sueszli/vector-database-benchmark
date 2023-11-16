"""Cluster Resolvers are used for dynamic cluster IP/hostname resolution."""
import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export

def format_master_url(master, rpc_layer=None):
    if False:
        print('Hello World!')
    if rpc_layer:
        return '%s://%s' % (rpc_layer, master)
    else:
        return master

def get_accelerator_devices(master, config_proto):
    if False:
        while True:
            i = 10
    'Returns accelerator devices given a master and a configuration.'
    if context.executing_eagerly():
        logical_devices = config.list_logical_devices()
        devices = []
        for d in logical_devices:
            if d.device_type == 'CPU' or d.device_type == 'XLA_CPU':
                continue
            devices.append(session._DeviceAttributes(d.name, d.device_type, 0, 0))
        return devices
    else:
        with ops.Graph().as_default():
            with session.Session(master, config=config_proto) as s:
                devices = s.list_devices()
        return devices

@tf_export('distribute.cluster_resolver.ClusterResolver')
@six.add_metaclass(abc.ABCMeta)
class ClusterResolver(object):
    """Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...) and gives TensorFlow necessary
  information to set up distributed training.

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors of `tf.distribute.cluster_resolver.ClusterResolver`
  subclass: In addition to these abstract methods, when task_type, task_id, and
  rpc_layer attributes are applicable, you should also implement them either as
  properties with getters or setters, or directly set the attributes
  `self._task_type`, `self._task_id`, or `self._rpc_layer` so the base class'
  getters and setters are used. See
  `tf.distribute.cluster_resolver.SimpleClusterResolver.__init__` for an
  example.

  In general, multi-client tf.distribute strategies such as
  `tf.distribute.experimental.MultiWorkerMirroredStrategy` require task_type and
  task_id properties to be available in the `ClusterResolver` they are using. On
  the other hand, these concepts are not applicable in single-client strategies,
  such as `tf.distribute.experimental.TPUStrategy`, because the program is only
  expected to be run on one task, so there should not be a need to have code
  branches according to task type and task id.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  """

    @abc.abstractmethod
    def cluster_spec(self):
        if False:
            print('Hello World!')
        'Retrieve the current state of the cluster and return a `tf.train.ClusterSpec`.\n\n    Returns:\n      A `tf.train.ClusterSpec` representing the state of the cluster at the\n      moment this function is called.\n\n    Implementors of this function must take care in ensuring that the\n    ClusterSpec returned is up-to-date at the time of calling this function.\n    This usually means retrieving the information from the underlying cluster\n    management system every time this function is invoked and reconstructing\n    a cluster_spec, rather than attempting to cache anything.\n    '
        raise NotImplementedError()

    @abc.abstractmethod
    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            for i in range(10):
                print('nop')
        'Retrieves the name or URL of the session master.\n\n    Note: this is only useful for TensorFlow 1.x.\n\n    Args:\n      task_type: (Optional) The type of the TensorFlow task of the master.\n      task_id: (Optional) The index of the TensorFlow task of the master.\n      rpc_layer: (Optional) The RPC protocol for the given cluster.\n\n    Returns:\n      The name or URL of the session master.\n\n    Implementors of this function must take care in ensuring that the master\n    returned is up-to-date at the time to calling this function. This usually\n    means retrieving the master every time this function is invoked.\n    '
        raise NotImplementedError()

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        if False:
            i = 10
            return i + 15
        'Returns the number of accelerator cores per worker.\n\n    This returns the number of accelerator cores (such as GPUs and TPUs)\n    available per worker.\n\n    Optionally, we allow callers to specify the task_type, and task_id, for\n    if they want to target a specific TensorFlow task to query\n    the number of accelerators. This is to support heterogenous environments,\n    where the number of accelerators cores per host is different.\n\n    Args:\n      task_type: (Optional) The type of the TensorFlow task of the machine we\n        want to query.\n      task_id: (Optional) The index of the TensorFlow task of the machine we\n        want to query.\n      config_proto: (Optional) Configuration for starting a new session to\n        query how many accelerator cores it has.\n\n    Returns:\n      A map of accelerator types to number of cores.\n    '
        master = self.master(task_type, task_id)
        devices = get_accelerator_devices(master, config_proto)
        mapping = collections.defaultdict(int)
        for device in devices:
            if task_type is not None and task_id is not None:
                job_path = '/job:%s' % task_type
                task_path = '/task:%s' % task_id
                if job_path not in device.name or task_path not in device.name:
                    continue
            mapping[device.device_type] += 1
        return mapping

    @property
    def environment(self):
        if False:
            i = 10
            return i + 15
        'Returns the current environment which TensorFlow is running in.\n\n    There are two possible return values, "google" (when TensorFlow is running\n    in a Google-internal environment) or an empty string (when TensorFlow is\n    running elsewhere).\n\n    If you are implementing a ClusterResolver that works in both the Google\n    environment and the open-source world (for instance, a TPU ClusterResolver\n    or similar), you will have to return the appropriate string depending on the\n    environment, which you will have to detect.\n\n    Otherwise, if you are implementing a ClusterResolver that will only work\n    in open-source TensorFlow, you do not need to implement this property.\n    '
        return ''

    @property
    def task_type(self):
        if False:
            while True:
                i = 10
        'Returns the task type this `ClusterResolver` indicates.\n\n    In TensorFlow distributed environment, each job may have an applicable\n    task type. Valid task types in TensorFlow include\n    \'chief\': a worker that is designated with more responsibility,\n    \'worker\': a regular worker for training/evaluation,\n    \'ps\': a parameter server, or\n    \'evaluator\': an evaluator that evaluates the checkpoints for metrics.\n\n    See [Multi-worker configuration](\n    https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#multi-worker_configuration)\n    for more information about \'chief\' and \'worker\' task type, which are most\n    commonly used.\n\n    Having access to such information is useful when user needs to run specific\n    code according to task types. For example,\n\n    ```python\n    cluster_spec = tf.train.ClusterSpec({\n        "ps": ["localhost:2222", "localhost:2223"],\n        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]\n    })\n\n    # SimpleClusterResolver is used here for illustration; other cluster\n    # resolvers may be used for other source of task type/id.\n    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",\n                                            task_id=1)\n\n    ...\n\n    if cluster_resolver.task_type == \'worker\':\n      # Perform something that\'s only applicable on workers. This block\n      # will run on this particular instance since we\'ve specified this task to\n      # be a worker in above cluster resolver.\n    elif cluster_resolver.task_type == \'ps\':\n      # Perform something that\'s only applicable on parameter servers. This\n      # block will not run on this particular instance.\n    ```\n\n    Returns `None` if such information is not available or is not applicable\n    in the current distributed environment, such as training with\n    `tf.distribute.experimental.TPUStrategy`.\n\n    For more information, please see\n    `tf.distribute.cluster_resolver.ClusterResolver`\'s class doc.\n    '
        return getattr(self, '_task_type', None)

    @property
    def task_id(self):
        if False:
            print('Hello World!')
        'Returns the task id this `ClusterResolver` indicates.\n\n    In TensorFlow distributed environment, each job may have an applicable\n    task id, which is the index of the instance within its task type. This is\n    useful when user needs to run specific code according to task index. For\n    example,\n\n    ```python\n    cluster_spec = tf.train.ClusterSpec({\n        "ps": ["localhost:2222", "localhost:2223"],\n        "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]\n    })\n\n    # SimpleClusterResolver is used here for illustration; other cluster\n    # resolvers may be used for other source of task type/id.\n    simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",\n                                            task_id=0)\n\n    ...\n\n    if cluster_resolver.task_type == \'worker\' and cluster_resolver.task_id == 0:\n      # Perform something that\'s only applicable on \'worker\' type, id 0. This\n      # block will run on this particular instance since we\'ve specified this\n      # task to be a \'worker\', id 0 in above cluster resolver.\n    else:\n      # Perform something that\'s only applicable on other ids. This block will\n      # not run on this particular instance.\n    ```\n\n    Returns `None` if such information is not available or is not applicable\n    in the current distributed environment, such as training with\n    `tf.distribute.cluster_resolver.TPUClusterResolver`.\n\n    For more information, please see\n    `tf.distribute.cluster_resolver.ClusterResolver`\'s class docstring.\n    '
        return getattr(self, '_task_id', None)

    @task_type.setter
    def task_type(self, task_type):
        if False:
            print('Hello World!')
        'Setter of `task_type` property. See `task_type` property doc.'
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        if False:
            print('Hello World!')
        'Setter of `task_id` property. See `task_type` property doc.'
        self._task_id = task_id

@tf_export('distribute.cluster_resolver.SimpleClusterResolver')
class SimpleClusterResolver(ClusterResolver):
    """Simple implementation of ClusterResolver that accepts all attributes.

  Please see the base class for documentation of arguments of its constructor.

  It is useful if you want to specify some or all attributes.

  Usage example with `tf.distribute.Strategy`:

    ```Python
    cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                               "worker1.example.com:2222"]})

    # On worker 0
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=0,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = SimpleClusterResolver(cluster, task_type="worker",
                                             task_id=1,
                                             num_accelerators={"GPU": 8},
                                             rpc_layer="grpc")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

    def __init__(self, cluster_spec, master='', task_type=None, task_id=None, environment='', num_accelerators=None, rpc_layer=None):
        if False:
            return 10
        'Creates a SimpleClusterResolver from a ClusterSpec.'
        super(SimpleClusterResolver, self).__init__()
        self._task_type = task_type
        self._task_id = task_id
        self._environment = environment
        self._num_accelerators = num_accelerators
        self._rpc_layer = rpc_layer
        if not isinstance(cluster_spec, ClusterSpec):
            raise TypeError('cluster_spec must be a `tf.train.ClusterSpec`.')
        self._cluster_spec = cluster_spec
        if not isinstance(master, str):
            raise TypeError('master must be a string.')
        self._master = master

    def cluster_spec(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the ClusterSpec passed into the constructor.'
        return self._cluster_spec

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            i = 10
            return i + 15
        'Returns the master address to use when creating a session.\n\n    Note: this is only useful for TensorFlow 1.x.\n\n    Args:\n      task_type: (Optional) The type of the TensorFlow task of the master.\n      task_id: (Optional) The index of the TensorFlow task of the master.\n      rpc_layer: (Optional) The RPC used by distributed TensorFlow.\n\n    Returns:\n      The name or URL of the session master.\n\n    If a task_type and task_id is given, this will override the `master`\n    string passed into the initialization function.\n    '
        if task_type is not None and task_id is not None:
            master = self.cluster_spec().task_address(task_type, task_id)
        else:
            master = self._master
        return format_master_url(master, rpc_layer=rpc_layer or self._rpc_layer)

    @property
    def task_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self._task_type

    @property
    def task_id(self):
        if False:
            while True:
                i = 10
        return self._task_id

    @task_type.setter
    def task_type(self, task_type):
        if False:
            return 10
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        if False:
            while True:
                i = 10
        self._task_id = task_id

    @property
    def environment(self):
        if False:
            for i in range(10):
                print('nop')
        return self._environment

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        if False:
            i = 10
            return i + 15
        'Returns the number of accelerator cores per worker.\n\n    The SimpleClusterResolver does not do automatic detection of accelerators,\n    and thus all arguments are unused and we simply return the value provided\n    in the constructor.\n\n    Args:\n      task_type: Unused.\n      task_id: Unused.\n      config_proto: Unused.\n    '
        del task_type, task_id, config_proto
        if self._num_accelerators is None:
            return {}
        return self._num_accelerators

    @property
    def rpc_layer(self):
        if False:
            i = 10
            return i + 15
        return self._rpc_layer

    @rpc_layer.setter
    def rpc_layer(self, rpc_layer):
        if False:
            print('Hello World!')
        self._rpc_layer = rpc_layer

@tf_export('distribute.cluster_resolver.UnionResolver')
class UnionClusterResolver(ClusterResolver):
    """Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional ClusterResolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.

  An example to combine two cluster resolvers:

    ```Python
    cluster_0 = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                                 "worker1.example.com:2222"]})
    cluster_resolver_0 = SimpleClusterResolver(cluster, task_type="worker",
                                               task_id=0,
                                               rpc_layer="grpc")

    cluster_1 = tf.train.ClusterSpec({"ps": ["ps0.example.com:2222",
                                             "ps1.example.com:2222"]})
    cluster_resolver_1 = SimpleClusterResolver(cluster, task_type="ps",
                                               task_id=0,
                                               rpc_layer="grpc")

    # Its task type would be "worker".
    cluster_resolver = UnionClusterResolver(cluster_resolver_0,
                                            cluster_resolver_1)
    ```

  An example to override the number of GPUs in a TFConfigClusterResolver
  instance:

    ```Python
    tf_config = TFConfigClusterResolver()
    gpu_override = SimpleClusterResolver(tf_config.cluster_spec(),
                                         num_accelerators={"GPU": 1})
    cluster_resolver = UnionResolver(gpu_override, tf_config)
    ```
  """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Initializes a UnionClusterResolver with other ClusterResolvers.\n\n    Args:\n      *args: `ClusterResolver` objects to be unionized.\n      **kwargs:\n        rpc_layer - (Optional) Override value for the RPC layer used by\n          TensorFlow.\n        task_type - (Optional) Override value for the current task type.\n        task_id - (Optional) Override value for the current task index.\n\n    Raises:\n      TypeError: If any argument is not a subclass of `ClusterResolvers`.\n      ValueError: If there are no arguments passed.\n    '
        super(UnionClusterResolver, self).__init__()
        self._rpc_layer = kwargs.pop('rpc_layer', None)
        self._task_type = kwargs.pop('task_type', None)
        self._task_id = kwargs.pop('task_id', None)
        if kwargs:
            raise ValueError('Unexpected kwargs provided {!r}'.format(kwargs))
        if not args:
            raise ValueError('At least one ClusterResolver is required.')
        for cluster_resolver in args:
            if not isinstance(cluster_resolver, ClusterResolver):
                raise TypeError('All arguments must be a sub-class of `ClusterResolver.`')
        self._cluster_resolvers = args

    def cluster_spec(self):
        if False:
            print('Hello World!')
        'Returns a union of all the ClusterSpecs from the ClusterResolvers.\n\n    Returns:\n      A ClusterSpec containing host information merged from all the underlying\n      ClusterResolvers.\n\n    Raises:\n      KeyError: If there are conflicting keys detected when merging two or\n      more dictionaries, this exception is raised.\n\n    Note: If there are multiple ClusterResolvers exposing ClusterSpecs with the\n    same job name, we will merge the list/dict of workers.\n\n    If *all* underlying ClusterSpecs expose the set of workers as lists, we will\n    concatenate the lists of workers, starting with the list of workers from\n    the first ClusterResolver passed into the constructor.\n\n    If *any* of the ClusterSpecs expose the set of workers as a dict, we will\n    treat all the sets of workers as dicts (even if they are returned as lists)\n    and will only merge them into a dict if there is no conflicting keys. If\n    there is a conflicting key, we will raise a `KeyError`.\n    '
        merged_cluster = {}
        for cluster_resolver in self._cluster_resolvers:
            cluster_spec = cluster_resolver.cluster_spec()
            cluster_dict = cluster_spec.as_dict()
            for (job_name, tasks) in cluster_dict.items():
                if job_name in merged_cluster:
                    if isinstance(tasks, dict):
                        merged_cluster[job_name] = {}
                elif isinstance(tasks, list):
                    merged_cluster[job_name] = []
                else:
                    merged_cluster[job_name] = {}
        for cluster_resolver in self._cluster_resolvers:
            cluster_spec = cluster_resolver.cluster_spec()
            cluster_dict = cluster_spec.as_dict()
            for (job_name, tasks) in cluster_dict.items():
                if isinstance(merged_cluster[job_name], list):
                    merged_cluster[job_name].extend(tasks)
                else:
                    if isinstance(tasks, list):
                        task_dict = dict(zip(range(0, len(tasks)), tasks))
                    else:
                        task_dict = tasks.copy()
                    task_keys = set(task_dict)
                    merged_keys = set(merged_cluster[job_name].keys())
                    intersected_keys = task_keys.intersection(merged_keys)
                    if intersected_keys:
                        raise KeyError('Duplicate keys detected when merging two ClusterSpecs: %s' % repr(intersected_keys))
                    merged_cluster[job_name].update(task_dict)
        return ClusterSpec(merged_cluster)

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            return 10
        'Returns the master address to use when creating a session.\n\n    This usually returns the master from the first ClusterResolver passed in,\n    but you can override this by specifying the task_type and task_id.\n\n    Note: this is only useful for TensorFlow 1.x.\n\n    Args:\n      task_type: (Optional) The type of the TensorFlow task of the master.\n      task_id: (Optional) The index of the TensorFlow task of the master.\n      rpc_layer: (Optional) The RPC protocol for the given cluster.\n\n    Returns:\n      The name or URL of the session master.\n    '
        if task_type is not None and task_id is not None:
            master = self.cluster_spec().task_address(task_type, task_id)
            return format_master_url(master, rpc_layer or self._rpc_layer)
        return self._cluster_resolvers[0].master(rpc_layer=rpc_layer)

    @property
    def task_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self._task_type or self._cluster_resolvers[0].task_type

    @property
    def task_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._task_id or self._cluster_resolvers[0].task_id

    @task_type.setter
    def task_type(self, task_type):
        if False:
            for i in range(10):
                print('nop')
        self._task_type = task_type

    @task_id.setter
    def task_id(self, task_id):
        if False:
            return 10
        self._task_id = task_id

    @property
    def environment(self):
        if False:
            return 10
        return self._cluster_resolvers[0].environment

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        if False:
            for i in range(10):
                print('nop')
        return self._cluster_resolvers[0].num_accelerators(task_type, task_id, config_proto)

    @property
    def rpc_layer(self):
        if False:
            print('Hello World!')
        return self._rpc_layer or self._cluster_resolvers[0].rpc_layer

    @rpc_layer.setter
    def rpc_layer(self, rpc_layer):
        if False:
            while True:
                i = 10
        self._rpc_layer = rpc_layer