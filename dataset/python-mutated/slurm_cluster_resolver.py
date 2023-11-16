"""Implementation of Cluster Resolvers for Slurm workload manager."""
import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export

def expand_hostlist(hostlist):
    if False:
        while True:
            i = 10
    "Create a list of hosts out of a SLURM hostlist.\n\n  The order of nodes is preserved and no deduplication is done\n  Input: 'n[1-2],m5,o[3-4,6,7-9]')\n  Output: ['n1', 'n2', 'm5', 'o3', 'o4', 'o6', 'o7', 'o8', 'o9']\n  "

    def split_hostlist(hostlist):
        if False:
            print('Hello World!')
        "Split hostlist at commas outside of range expressions ('[3-5]')."
        in_brackets = False
        cur_host = ''
        for c in hostlist:
            if in_brackets:
                assert c != '['
                if c == ']':
                    in_brackets = False
            elif c == '[':
                in_brackets = True
            elif c == ',':
                assert cur_host != ''
                yield cur_host
                cur_host = ''
                continue
            cur_host += c
        if cur_host:
            yield cur_host

    def expand_range_expression(range_exp):
        if False:
            while True:
                i = 10
        "Expand a range expression like '3-5' to values 3,4,5."
        for part in range_exp.split(','):
            sub_range = part.split('-')
            if len(sub_range) == 1:
                sub_range = sub_range * 2
            else:
                assert len(sub_range) == 2
            num_digits = len(sub_range[0])
            for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
                yield str(i).zfill(num_digits)
    hosts = []
    try:
        for part in split_hostlist(hostlist):
            m = re.match('([^,[\\]]*)(\\[([^\\]]+)\\])?$', part)
            if m is None:
                raise ValueError('Invalid part: %s' % part)
            prefix = m.group(1) or ''
            if m.group(3) is None:
                hosts.append(prefix)
            else:
                hosts.extend((prefix + i for i in expand_range_expression(m.group(3))))
    except Exception as e:
        raise ValueError('Invalid hostlist format "%s": %s' % (hostlist, e))
    return hosts

def expand_tasks_per_node(tasks_per_node):
    if False:
        print('Hello World!')
    "Expands the tasks per node expression from SLURM.\n\n  The order is preserved so it can be matched to the hostlist\n  Input: '3(x2),2,1'\n  Output: [3, 3, 2, 1]\n  "
    result = []
    try:
        for part in tasks_per_node.split(','):
            m = re.match('(\\d+)(\\(x(\\d+)\\))?$', part)
            assert m is not None
            num_tasks = int(m.group(1))
            num_repetitions = int(m.group(3) or 1)
            result.extend([num_tasks] * num_repetitions)
    except Exception as e:
        raise ValueError('Invalid tasks-per-node list format "%s": %s' % (tasks_per_node, e))
    return result

def _get_slurm_var(name):
    if False:
        print('Hello World!')
    'Gets the SLURM variable from the environment.\n\n  Args:\n    name: Name of the step variable\n\n  Returns:\n    SLURM_<name> from os.environ\n  Raises:\n    RuntimeError if variable is not found\n  '
    name = 'SLURM_' + name
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError('%s not found in environment. Not running inside a SLURM step?' % name)

def _get_num_slurm_tasks():
    if False:
        while True:
            i = 10
    'Returns the number of SLURM tasks of the current job step.\n\n  Returns:\n    The number of tasks as an int\n  '
    return int(_get_slurm_var('STEP_NUM_TASKS'))

def _get_num_nvidia_gpus():
    if False:
        for i in range(10):
            print('nop')
    'Gets the number of NVIDIA GPUs by using CUDA_VISIBLE_DEVICES and nvidia-smi.\n\n  Returns:\n    Number of GPUs available on the node\n  Raises:\n    RuntimeError if executing nvidia-smi failed\n  '
    try:
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except KeyError:
        pass
    try:
        output = subprocess.check_output(['nvidia-smi', '--list-gpus'], encoding='utf-8')
        return sum((l.startswith('GPU ') for l in output.strip().split('\n')))
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Could not get number of GPUs from nvidia-smi. Maybe it is missing?\nOutput: %s' % e.output)

def get_num_gpus():
    if False:
        print('Hello World!')
    'Returns the number of GPUs visible on the current node.\n\n  Currently only implemented for NVIDIA GPUs.\n  '
    return _get_num_nvidia_gpus()

@tf_export('distribute.cluster_resolver.SlurmClusterResolver')
class SlurmClusterResolver(ClusterResolver):
    """ClusterResolver for system with Slurm workload manager.

  This is an implementation of ClusterResolver for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number
  of GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  used for distributed TensorFlow.
  """

    def __init__(self, jobs=None, port_base=8888, gpus_per_node=None, gpus_per_task=None, tasks_per_node=None, auto_set_gpu=True, rpc_layer='grpc'):
        if False:
            return 10
        "Creates a new SlurmClusterResolver object.\n\n    For any parameter not set it will query the environment for the value.\n    It uses those parameters to check which nodes have processes reside on and\n    resolves their hostnames.\n    With the number tasks per node it offsets the port number for each process.\n    With the number of GPUs per node and per task it allocates GPUs to tasks by\n    setting environment variables.\n    Using the resolver works best (and is easier) with homogeneous tasks but\n    heterogeneous tasks (number of tasks varying per node) are also possible as\n    long as the number of GPUs per task stays constant.\n\n    Used environment variables:\n      - SLURM_PROCID\n      - (opt) SLURM_STEP_NUM_TASKS\n      - (opt) SLURM_STEP_NODELIST\n      - (opt) SLURM_STEP_TASKS_PER_NODE\n\n    Args:\n      jobs: Dictionary with job names as key and number of tasks in the job as\n        value. Defaults to as many 'worker's as there are (Slurm) tasks.\n      port_base: The first port number to start with for processes on a node.\n      gpus_per_node: Number of GPUs available on each node. Defaults to the\n        number of GPUs reported by nvidia-smi\n      gpus_per_task: Number of GPUs to be used for each task. Default is to\n        evenly distribute the gpus_per_node to tasks_per_node.\n      tasks_per_node: Number of tasks running on each node. Can be an integer if\n        the number of tasks per node is constant or a dictionary mapping\n        hostnames to number of tasks on that node. If not set the Slurm\n        environment is queried for the correct mapping.\n      auto_set_gpu: Set the visible CUDA devices automatically while resolving\n        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.\n        Defaults to True.\n      rpc_layer: The protocol TensorFlow used to communicate between nodes.\n        Defaults to 'grpc'.\n\n    Returns:\n      A ClusterResolver object which can be used with distributed TensorFlow.\n\n    Raises:\n      RuntimeError: If requested more GPUs per node than available or\n        requested more tasks than assigned tasks or\n        resolving missing values from the environment failed.\n    "
        self._rank = self._resolve_own_rank()
        if jobs is None:
            jobs = {'worker': self._resolve_num_tasks()}
        self._jobs = jobs
        self._port_base = port_base
        if tasks_per_node is None:
            self._task_configuration = self._resolve_task_configuration()
        elif isinstance(tasks_per_node, dict):
            self._task_configuration = tasks_per_node
        else:
            hostlist = self._resolve_hostlist()
            self._task_configuration = {host: int(tasks_per_node) for host in hostlist}
        max_tasks_per_node = max(self._task_configuration.values())
        num_tasks = sum(self._task_configuration.values())
        if gpus_per_node is None:
            gpus_per_node = get_num_gpus()
        if gpus_per_task is None:
            gpus_per_task = gpus_per_node // max_tasks_per_node
        self._gpus_per_node = gpus_per_node
        self._gpus_per_task = gpus_per_task
        self._auto_set_gpu = auto_set_gpu
        self.task_type = None
        self.task_id = None
        self.rpc_layer = rpc_layer
        self._gpu_allocation = []
        self._cluster_allocation = {}
        if max_tasks_per_node * self._gpus_per_task > self._gpus_per_node:
            raise RuntimeError('Requested more GPUs per node than available.')
        if sum(self._jobs.values()) != num_tasks:
            raise RuntimeError('Requested {} tasks but only {} were assigned.'.format(sum(self._jobs.values()), num_tasks))

    def _resolve_own_rank(self):
        if False:
            return 10
        'Returns the rank of the current task in range [0, num_tasks).'
        return int(_get_slurm_var('PROCID'))

    def _resolve_num_tasks(self):
        if False:
            print('Hello World!')
        'Returns the number of tasks for the current job step.'
        return _get_num_slurm_tasks()

    def _resolve_hostlist(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of hostnames for nodes running the current job step.'
        return expand_hostlist(_get_slurm_var('STEP_NODELIST'))

    def _resolve_task_configuration(self):
        if False:
            return 10
        'Creates a mapping of hostnames to the number of tasks allocated on it.\n\n    Reads the SLURM environment to determine the nodes involved in the current\n    job step and number of tasks running on each node.\n\n    Returns a dictionary mapping each hostname to the number of tasks.\n    '
        hostlist = self._resolve_hostlist()
        tasks_per_node = expand_tasks_per_node(_get_slurm_var('STEP_TASKS_PER_NODE'))
        return {host: num_tasks for (host, num_tasks) in zip(hostlist, tasks_per_node)}

    def cluster_spec(self):
        if False:
            print('Hello World!')
        "Returns a ClusterSpec object based on the latest instance group info.\n\n    This returns a ClusterSpec object for use based on information from the\n    specified initialization parameters and Slurm environment variables. The\n    cluster specification is resolved each time this function is called. The\n    resolver extract hostnames of nodes by scontrol and pack tasks in that\n    order until a node a has number of tasks that is equal to specification.\n    GPUs on nodes are allocated to tasks by specification through setting\n    CUDA_VISIBLE_DEVICES environment variable.\n\n    Returns:\n      A ClusterSpec containing host information retrieved from Slurm's\n        environment variables.\n    "
        task_list = []
        self._gpu_allocation = []
        self._cluster_allocation = {}
        for (host, num_tasks) in sorted(self._task_configuration.items()):
            for (port_offset, gpu_offset) in zip(range(num_tasks), range(0, self._gpus_per_node, self._gpus_per_task)):
                host_addr = '%s:%d' % (host, self._port_base + port_offset)
                task_list.append(host_addr)
                gpu_id_list = []
                for gpu_id in range(gpu_offset, gpu_offset + self._gpus_per_task):
                    gpu_id_list.append(str(gpu_id))
                self._gpu_allocation.append(','.join(gpu_id_list))
        cluster_rank_offset_start = 0
        cluster_rank_offset_end = 0
        for (task_type, num_tasks) in sorted(self._jobs.items()):
            cluster_rank_offset_end = cluster_rank_offset_start + num_tasks
            self._cluster_allocation[task_type] = task_list[cluster_rank_offset_start:cluster_rank_offset_end]
            if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
                self.task_type = task_type
                self.task_id = self._rank - cluster_rank_offset_start
            cluster_rank_offset_start = cluster_rank_offset_end
        if self._auto_set_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_allocation[self._rank]
        return ClusterSpec(self._cluster_allocation)

    def get_task_info(self):
        if False:
            while True:
                i = 10
        'Returns job name and task_id for the process which calls this.\n\n    This returns the job name and task index for the process which calls this\n    function according to its rank and cluster specification. The job name and\n    task index are set after a cluster is constructed by cluster_spec otherwise\n    defaults to None.\n\n    Returns:\n      A string specifying job name the process belongs to and an integer\n        specifying the task index the process belongs to in that job.\n    '
        return (self.task_type, self.task_id)

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        if False:
            while True:
                i = 10
        'Returns the master string for connecting to a TensorFlow master.\n\n    Args:\n      task_type: (Optional) Overrides the default auto-selected task type.\n      task_id: (Optional) Overrides the default auto-selected task index.\n      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses\n        to communicate across nodes.\n\n    Returns:\n      A connection string for connecting to a TensorFlow master.\n    '
        task_type = task_type if task_type is not None else self.task_type
        task_id = task_id if task_id is not None else self.task_id
        if task_type is not None and task_id is not None:
            return format_master_url(self.cluster_spec().task_address(task_type, task_id), rpc_layer or self.rpc_layer)
        return ''

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        if False:
            print('Hello World!')
        del task_type, task_id, config_proto
        return {'GPU': self._gpus_per_task}