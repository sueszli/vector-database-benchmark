import contextlib
import math
import os
import warnings
try:
    import optuna
    _optuna_available = True
except ImportError:
    _optuna_available = False
from cupy._core import _optimize_config
from cupyx import profiler

def _optimize(optimize_config, target_func, suggest_func, default_best, ignore_error=()):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(optimize_config, _optimize_config._OptimizationConfig)
    assert callable(target_func)
    assert callable(suggest_func)

    def objective(trial):
        if False:
            return 10
        args = suggest_func(trial)
        max_total_time = optimize_config.max_total_time_per_trial
        try:
            perf = profiler.benchmark(target_func, args, max_duration=max_total_time)
            return perf.gpu_times.mean()
        except Exception as e:
            if isinstance(e, ignore_error):
                return math.inf
            else:
                raise e
    study = optuna.create_study()
    study.enqueue_trial(default_best)
    study.optimize(objective, n_trials=optimize_config.max_trials, timeout=optimize_config.timeout)
    return study.best_trial

@contextlib.contextmanager
def optimize(*, key=None, path=None, readonly=False, **config_dict):
    if False:
        print('Hello World!')
    "Context manager that optimizes kernel launch parameters.\n\n    In this context, CuPy's routines find the best kernel launch parameter\n    values (e.g., the number of threads and blocks). The found values are\n    cached and reused with keys as the shapes, strides and dtypes of the\n    given inputs arrays.\n\n    Args:\n        key (string or None): The cache key of optimizations.\n        path (string or None): The path to save optimization cache records.\n            When path is specified and exists, records will be loaded from\n            the path. When readonly option is set to ``False``, optimization\n            cache records will be saved to the path after the optimization.\n        readonly (bool): See the description of ``path`` option.\n        max_trials (int): The number of trials that defaults to 100.\n        timeout (float):\n            Stops study after the given number of seconds. Default is 1.\n        max_total_time_per_trial (float):\n            Repeats measuring the execution time of the routine for the\n            given number of seconds. Default is 0.1.\n\n    Examples\n    --------\n    >>> import cupy\n    >>> from cupyx import optimizing\n    >>>\n    >>> x = cupy.arange(100)\n    >>> with optimizing.optimize():\n    ...     cupy.sum(x)\n    ...\n    array(4950)\n\n    .. note::\n      Optuna (https://optuna.org) installation is required.\n      Currently it works for reduction operations only.\n    "
    if not _optuna_available:
        raise RuntimeError('Optuna is required to run optimization. See https://optuna.org/ for the installation instructions.')
    old_context = _optimize_config.get_current_context()
    context = _optimize_config.get_new_context(key, _optimize, config_dict)
    _optimize_config.set_current_context(context)
    if path is not None:
        if os.path.exists(path):
            context.load(path)
        elif readonly:
            warnings.warn('\nThe specified path {} could not be found, and the readonly option is set.\nThe optimization results will never be stored.\n'.format(path))
    try:
        yield context
        if path is not None and (not readonly):
            if context._is_dirty() or not os.path.exists(path):
                context.save(path)
    finally:
        _optimize_config.set_current_context(old_context)