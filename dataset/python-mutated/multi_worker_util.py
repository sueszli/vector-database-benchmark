"""Utilities for multi-worker distribution strategies."""
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib

def normalize_cluster_spec(cluster_spec):
    if False:
        return 10
    'Makes `cluster_spec` into a `ClusterSpec` object.\n\n  Args:\n    cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the\n      cluster configurations.\n\n  Returns:\n    a `ClusterSpec` object.\n\n  Raises:\n    ValueError: if `cluster_spec` is not a dict or a `ClusterSpec` or a\n      `ClusterDef`.\n  '
    if isinstance(cluster_spec, (dict, cluster_pb2.ClusterDef)):
        return server_lib.ClusterSpec(cluster_spec)
    elif not isinstance(cluster_spec, server_lib.ClusterSpec):
        raise ValueError("`cluster_spec' should be dict or a `tf.train.ClusterSpec` or a `tf.train.ClusterDef` object")
    return cluster_spec

def task_count(cluster_spec, task_type):
    if False:
        i = 10
        return i + 15
    try:
        return cluster_spec.num_tasks(task_type)
    except ValueError:
        return 0

def _validate_cluster_spec(cluster_spec, task_type, task_id):
    if False:
        while True:
            i = 10
    'Validates `cluster_spec`.\n\n  It checks:\n  1) task type is one of "chief", "worker", "ps", "evaluator", or not provided\n     (None).\n  2) whether there is such a task type as `task_type` in the `cluster_spec`. The\n     only exception is `evaluator`. In other words, it is still a valid\n     configuration when `task_type` is `evaluator` but it doesn\'t appear in\n     `cluster_spec`. This is to be compatible with `TF_CONFIG` in Estimator.\n  3) whether there is at most one "chief" job.\n  4) whether there is at most one "evaluator" job.\n  5) whether the `task_id` is smaller than the number of tasks for that\n     particular `task_type`.\n\n  Args:\n    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.\n    task_type: string indicating the type of the task.\n    task_id: the id of the `task_type` in this cluster.\n\n  Raises:\n    ValueError: if `cluster_spec` fails any check.\n  '
    allowed_task_types = ('chief', 'worker', 'evaluator', 'ps', None)
    cluster_spec = normalize_cluster_spec(cluster_spec)
    if any((job not in allowed_task_types for job in cluster_spec.jobs)):
        raise ValueError('Disallowed task type found in cluster spec. Allowed types are {} and the cluster spec is {}.'.format(allowed_task_types, cluster_spec))
    if task_type not in allowed_task_types:
        raise ValueError('Unrecognized task_type: {}, valid task types are: {}'.format(task_type, allowed_task_types))
    if task_type and task_type not in cluster_spec.jobs and (task_type != 'evaluator'):
        raise ValueError('`task_type` %r not found in cluster_spec.' % task_type)
    if task_count(cluster_spec, 'chief') > 1:
        raise ValueError("There must be at most one 'chief' job.")
    if task_count(cluster_spec, 'evaluator') > 1:
        raise ValueError("There must be at most one 'evaluator' job.")
    if task_type in cluster_spec.jobs and task_id >= task_count(cluster_spec, task_type):
        raise ValueError('The `task_id` %d exceeds the maximum id of %s.' % (task_id, task_type))

def is_chief(cluster_spec=None, task_type=None, task_id=None):
    if False:
        while True:
            i = 10
    'Returns whether the given task is chief in the cluster.\n\n  Since there is at most one evaluator and the evaluator itself should be\n  independent of the training cluster, the evaluator job is also a chief job on\n  its own.\n\n  If this is currently running under a `_WorkerContext` of distribute\n  coordinator, the arguments can be omitted as the result is already available.\n\n  Args:\n    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object specifying the\n      cluster configurations.\n    task_type: the task type in the cluster.\n    task_id: the task id in the cluster.\n\n  Returns:\n    a boolean indicating whether the given task is chief.\n\n  Raises:\n    ValueError: if `task_type` is not in the `cluster_spec` or `task_id` exceeds\n      the maximum id of the `task_type`.\n  '
    if has_worker_context():
        return dc_context.get_current_worker_context().is_chief
    _validate_cluster_spec(cluster_spec, task_type, task_id)
    cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type == 'chief' or task_type == 'evaluator':
        return True
    if 'chief' not in cluster_spec and task_type == 'worker' and (task_id == 0):
        return True
    return False

def collective_leader(cluster_spec, task_type, task_id):
    if False:
        i = 10
        return i + 15
    'Return the job name for the leader of for collective ops.\n\n  Args:\n    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object specifying the\n      cluster configurations.\n    task_type: the task type in the cluster.\n    task_id: the task id in the cluster.\n\n  Returns:\n    a string indicating the leader job name or empty string if no need to set\n    leader job.\n  '
    cluster_spec = normalize_cluster_spec(cluster_spec)
    if not cluster_spec.as_dict():
        return ''
    _validate_cluster_spec(cluster_spec, task_type, task_id)
    if task_type == 'evaluator':
        return ''
    if 'chief' in cluster_spec.jobs:
        return '/job:chief/replica:0/task:0'
    assert 'worker' in cluster_spec.jobs
    return '/job:worker/replica:0/task:0'

def coordination_leader(cluster_spec):
    if False:
        return 10
    'Return the task name of the coordination service leader.\n\n  Args:\n    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object sxpecifying the\n      cluster configurations.\n\n  Returns:\n    a string indicating the task name of the coordination service leader.\n  '
    cluster_spec = normalize_cluster_spec(cluster_spec)
    if not cluster_spec.as_dict():
        return ''
    if 'ps' in cluster_spec.jobs:
        return '/job:ps/replica:0/task:0'
    if 'chief' in cluster_spec.jobs:
        return '/job:chief/replica:0/task:0'
    assert 'worker' in cluster_spec.jobs
    return '/job:worker/replica:0/task:0'

def worker_count(cluster_spec, task_type):
    if False:
        print('Hello World!')
    'Returns the number of workers in the cluster.'
    _validate_cluster_spec(cluster_spec, task_type, task_id=0)
    cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type not in ['chief', 'worker', 'evaluator']:
        raise ValueError('Unexpected `task_type` %r' % task_type)
    if task_type == 'evaluator':
        return len(cluster_spec['evaluator'])
    else:
        return len(cluster_spec.get('chief', [])) + len(cluster_spec.get('worker', []))

def id_in_cluster(cluster_spec, task_type, task_id):
    if False:
        while True:
            i = 10
    'Returns a unique id for the task in the `task_type`\'s cluster.\n\n  It returns an id ranging from [0, `worker_count(task_type, task_id)`).\n\n  Note: this function assumes that "evaluate" job is in its own cluster or its\n  own partition of a cluster.\n\n  Args:\n    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.\n    task_type: string indicating the type of the task.\n    task_id: the id of the `task_type` in this cluster.\n\n  Returns:\n    an int indicating the unique id.\n\n  Throws:\n    ValueError: if `task_type` is not "chief", "worker" or "evaluator".\n  '
    _validate_cluster_spec(cluster_spec, task_type, task_id)
    cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type == 'chief':
        return 0
    if task_type == 'worker':
        return task_id + len(cluster_spec.get('chief', []))
    if task_type == 'evaluator':
        return task_id
    raise ValueError('There is no id for task_type %r' % task_type)

def should_save_checkpoint():
    if False:
        print('Hello World!')
    'Returns whether the current worker should save checkpoints.\n\n  In multi-worker training, if saving checkpoint is requested by user, or needed\n  for fault-tolerance, the cluster should save checkpoint but not necessarily\n  every worker in the cluster should.\n\n  TODO(rchao): Consider generalizing this util to be `should_save_file` as there\n  can be other files to save such as summary.\n\n  Returns:\n      Whether this particular worker in the cluster should save checkpoints.\n  '
    return dc_context.get_current_worker_context().should_checkpoint

def should_load_checkpoint():
    if False:
        i = 10
        return i + 15
    'Returns whether the current worker should load checkpoints.\n\n  In multi-worker training, if loading checkpoint is requested by user, or\n  needed for fault-tolerance, the cluster should load checkpoint but not\n  necessarily every worker in the cluster should.\n\n  Returns:\n      Whether this particular worker in the cluster should load checkpoints.\n  '
    return dc_context.get_current_worker_context().experimental_should_init

def wait_for_other_workers():
    if False:
        for i in range(10):
            print('nop')
    'Waits for other workers to reach the same call to this method.'
    return dc_context.get_current_worker_context().wait_for_other_workers()

def has_worker_context():
    if False:
        i = 10
        return i + 15
    'Returns whether a worker context has been entered.'
    return dc_context.get_current_worker_context() is not None