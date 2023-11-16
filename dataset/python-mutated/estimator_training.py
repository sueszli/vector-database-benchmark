"""Training utilities for Estimator to use Distribute Coordinator."""
import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
CHIEF = dc._TaskType.CHIEF
EVALUATOR = dc._TaskType.EVALUATOR
PS = dc._TaskType.PS
WORKER = dc._TaskType.WORKER

def _count_ps(cluster_spec):
    if False:
        while True:
            i = 10
    'Counts the number of parameter servers in cluster_spec.'
    if not cluster_spec:
        raise RuntimeError('Internal error: `_count_ps` does not expect empty cluster_spec.')
    return len(cluster_spec.as_dict().get(PS, []))

def _count_worker(cluster_spec, chief_task_type):
    if False:
        for i in range(10):
            print('nop')
    'Counts the number of workers (including chief) in cluster_spec.'
    if not cluster_spec:
        raise RuntimeError('Internal error: `_count_worker` does not expect empty cluster_spec.')
    return len(cluster_spec.as_dict().get(WORKER, [])) + len(cluster_spec.as_dict().get(chief_task_type, []))

def _get_global_id(cluster_spec, task_type, task_id, chief_task_type):
    if False:
        i = 10
        return i + 15
    'Returns the global id of the given task type in a cluster.'
    if not task_type:
        return 0
    task_type_ordered_list = []
    if chief_task_type in cluster_spec.jobs:
        task_type_ordered_list = [chief_task_type]
    task_type_ordered_list.extend([t for t in sorted(cluster_spec.jobs) if t != chief_task_type and t != PS])
    if PS in cluster_spec.jobs:
        task_type_ordered_list.append(PS)
    next_global_id = 0
    for t in task_type_ordered_list:
        if t == task_type:
            return next_global_id + task_id
        next_global_id += len(cluster_spec.job_tasks(t))
    raise RuntimeError('Internal Error: `task_type` ({}) is not in cluster_spec ({}).'.format(task_type, cluster_spec))

def _init_run_config_from_worker_context(config, worker_context):
    if False:
        print('Hello World!')
    "Initializes run config from distribute coordinator's worker context."
    config._service = None
    config._cluster_spec = worker_context.cluster_spec
    config._task_type = worker_context.task_type
    config._task_id = worker_context.task_id
    config._evaluation_master = worker_context.master_target
    config._master = worker_context.master_target
    config._is_chief = worker_context.is_chief
    if config._cluster_spec:
        if config._task_type != EVALUATOR:
            config._num_ps_replicas = _count_ps(config._cluster_spec)
            config._num_worker_replicas = _count_worker(config._cluster_spec, chief_task_type=CHIEF)
            config._global_id_in_cluster = _get_global_id(config._cluster_spec, config._task_type, config._task_id, chief_task_type=CHIEF)
        else:
            config._cluster_spec = server_lib.ClusterSpec({})
            config._num_ps_replicas = 0
            config._num_worker_replicas = 0
            config._global_id_in_cluster = None
    else:
        config._global_id_in_cluster = 0
        config._num_ps_replicas = 0
        config._num_worker_replicas = 1

def init_run_config(config, tf_config):
    if False:
        print('Hello World!')
    'Initializes RunConfig for distribution strategies.'
    if config._experimental_distribute and config._experimental_distribute.train_distribute:
        if config._train_distribute:
            raise ValueError('Either `train_distribute` or`experimental_distribute.train_distribute` can be set.')
        config._train_distribute = config._experimental_distribute.train_distribute
    if config._experimental_distribute and config._experimental_distribute.eval_distribute:
        if config._eval_distribute:
            raise ValueError('Either `eval_distribute` or`experimental_distribute.eval_distribute` can be set.')
        config._eval_distribute = config._experimental_distribute.eval_distribute
    cluster_spec = server_lib.ClusterSpec(tf_config.get('cluster', {}))
    config._init_distributed_setting_from_environment_var({})
    if config._train_distribute and config._experimental_distribute and config._experimental_distribute.remote_cluster:
        if cluster_spec:
            raise ValueError('Cannot set both "cluster_spec" of TF_CONFIG and `experimental_distribute.remote_cluster`')
        config._distribute_coordinator_mode = dc.CoordinatorMode.STANDALONE_CLIENT
        config._cluster_spec = config._experimental_distribute.remote_cluster
        logging.info('RunConfig initialized for Distribute Coordinator with STANDALONE_CLIENT mode')
        return
    if not cluster_spec or 'master' in cluster_spec.jobs or (not config._train_distribute):
        config._distribute_coordinator_mode = None
        config._init_distributed_setting_from_environment_var(tf_config)
        config._maybe_overwrite_session_config_for_distributed_training()
        logging.info('Not using Distribute Coordinator.')
        return
    assert tf_config
    config._cluster_spec = cluster_spec
    config._distribute_coordinator_mode = dc.CoordinatorMode.INDEPENDENT_WORKER
    logging.info('RunConfig initialized for Distribute Coordinator with INDEPENDENT_WORKER mode')

def should_run_distribute_coordinator(config):
    if False:
        return 10
    'Checks the config to see whether to run distribute coordinator.'
    if not hasattr(config, '_distribute_coordinator_mode') or config._distribute_coordinator_mode is None:
        logging.info('Not using Distribute Coordinator.')
        return False
    if not isinstance(config._distribute_coordinator_mode, six.string_types) or config._distribute_coordinator_mode not in [dc.CoordinatorMode.STANDALONE_CLIENT, dc.CoordinatorMode.INDEPENDENT_WORKER]:
        logging.warning('Unexpected distribute_coordinator_mode: %r', config._distribute_coordinator_mode)
        return False
    if not config.cluster_spec:
        logging.warning('Running `train_and_evaluate` locally, ignoring `experimental_distribute_coordinator_mode`.')
        return False
    return True

def train_and_evaluate(estimator, train_spec, eval_spec, executor_cls):
    if False:
        print('Hello World!')
    "Run distribute coordinator for Estimator's `train_and_evaluate`.\n\n  Args:\n    estimator: An `Estimator` instance to train and evaluate.\n    train_spec: A `TrainSpec` instance to specify the training specification.\n    eval_spec: A `EvalSpec` instance to specify the evaluation and export\n      specification.\n    executor_cls: the evaluation executor class of Estimator.\n\n  Raises:\n    ValueError: if `distribute_coordinator_mode` is None in RunConfig.\n  "
    run_config = estimator.config
    if not run_config._distribute_coordinator_mode:
        raise ValueError('Distribute coordinator mode is not specified in `RunConfig`.')

    def _worker_fn(strategy):
        if False:
            return 10
        'Function for worker task.'
        local_estimator = copy.deepcopy(estimator)
        local_estimator._config._train_distribute = strategy
        context = dc_context.get_current_worker_context()
        _init_run_config_from_worker_context(local_estimator._config, context)
        logging.info('Updated config: %s', str(vars(local_estimator._config)))
        local_estimator._train_distribution = strategy
        if run_config._distribute_coordinator_mode == dc.CoordinatorMode.INDEPENDENT_WORKER or context.is_chief:
            hooks = list(train_spec.hooks)
        else:
            hooks = []
        local_estimator._config._distribute_coordinator_mode = None
        local_estimator.train(input_fn=train_spec.input_fn, max_steps=train_spec.max_steps, hooks=hooks)

    def _eval_fn(strategy):
        if False:
            while True:
                i = 10
        'Function for evaluator task.'
        local_estimator = copy.deepcopy(estimator)
        local_estimator._config._eval_distribute = strategy
        _init_run_config_from_worker_context(local_estimator._config, dc_context.get_current_worker_context())
        logging.info('Updated config: %s', str(vars(local_estimator._config)))
        local_estimator._eval_distribution = strategy
        local_estimator._config._distribute_coordinator_mode = None
        executor = executor_cls(local_estimator, train_spec, eval_spec)
        executor._start_continuous_evaluation()
    if run_config._distribute_coordinator_mode == dc.CoordinatorMode.STANDALONE_CLIENT:
        cluster_spec = run_config.cluster_spec
        assert cluster_spec
    else:
        cluster_spec = None
    dc.run_distribute_coordinator(_worker_fn, run_config.train_distribute, _eval_fn, run_config.eval_distribute, mode=run_config._distribute_coordinator_mode, cluster_spec=cluster_spec, session_config=run_config.session_config)

def estimator_train(estimator, train_distributed_fn, hooks):
    if False:
        print('Hello World!')
    "Run distribute coordinator for Estimator's `train` method."
    assert estimator._config._distribute_coordinator_mode
    run_config = estimator._config
    assert estimator._config.cluster_spec
    cluster_spec = multi_worker_util.normalize_cluster_spec(estimator._config.cluster_spec)
    assert estimator._config._train_distribute
    if 'evaluator' in cluster_spec.jobs:
        raise ValueError("'evaluator' job is not supported if you don't use `train_and_evaluate`")
    if estimator._config._distribute_coordinator_mode != dc.CoordinatorMode.STANDALONE_CLIENT:
        raise ValueError('Only `STANDALONE_CLIENT` mode is supported when you call `estimator.train`')
    if estimator._config._train_distribute.extended.experimental_between_graph:
        raise ValueError('`Estimator.train` API is not supported for %s with `STANDALONE_CLIENT` mode.' % estimator._config._train_distribute.__class__.__name__)

    def _worker_fn(strategy):
        if False:
            i = 10
            return i + 15
        'Function for worker task.'
        local_estimator = copy.deepcopy(estimator)
        local_estimator._config._train_distribute = strategy
        context = dc_context.get_current_worker_context()
        _init_run_config_from_worker_context(local_estimator._config, context)
        logging.info('Updated config: %s', str(vars(local_estimator._config)))
        local_estimator._train_distribution = strategy
        if context.is_chief:
            chief_hooks = hooks
        else:
            chief_hooks = []
        train_distributed_fn(local_estimator, strategy, chief_hooks)
        return local_estimator
    return dc.run_distribute_coordinator(_worker_fn, estimator._config.train_distribute, mode=run_config._distribute_coordinator_mode, cluster_spec=cluster_spec, session_config=run_config.session_config)

def estimator_evaluate(estimator, evaluate_distributed_fn, hooks):
    if False:
        while True:
            i = 10
    "Run distribute coordinator for Estimator's `evaluate` method."
    assert estimator._config._distribute_coordinator_mode
    run_config = estimator._config
    assert estimator._config.cluster_spec
    cluster_spec = multi_worker_util.normalize_cluster_spec(estimator._config.cluster_spec)
    assert estimator._config._eval_distribute
    if 'evaluator' in cluster_spec.jobs:
        raise ValueError("'evaluator' job is not supported if you don't use `train_and_evaluate`")
    if estimator._config._distribute_coordinator_mode != dc.CoordinatorMode.STANDALONE_CLIENT:
        raise ValueError('Only `STANDALONE_CLIENT` mode is supported when you call `Estimator.evaluate`')
    if estimator._config._eval_distribute.extended.experimental_between_graph:
        raise ValueError('`Estimator.evaluate` API is not supported for %s with `STANDALONE_CLIENT` mode.' % estimator._config._eval_distribute.__class__.__name__)

    def _worker_fn(strategy):
        if False:
            i = 10
            return i + 15
        'Function for evaluation.'
        local_estimator = copy.deepcopy(estimator)
        local_estimator._config._eval_distribute = strategy
        context = dc_context.get_current_worker_context()
        _init_run_config_from_worker_context(local_estimator._config, context)
        logging.info('Updated config: %s', str(vars(local_estimator._config)))
        local_estimator._eval_distribution = strategy
        if context.is_chief:
            chief_hooks = hooks
        else:
            chief_hooks = []
        return evaluate_distributed_fn(local_estimator, strategy, chief_hooks)
    return dc.run_distribute_coordinator(_worker_fn, estimator._config.eval_distribute, mode=run_config._distribute_coordinator_mode, cluster_spec=cluster_spec, session_config=run_config.session_config)