from __future__ import annotations
from datetime import datetime
from typing import Any, Callable, Collection, Mapping, TypeVar
from airflow import settings
from airflow.utils.context import Context, lazy_mapping_from_context
R = TypeVar('R')
DEFAULT_FORMAT_PREFIX = 'airflow.ctx.'
ENV_VAR_FORMAT_PREFIX = 'AIRFLOW_CTX_'
AIRFLOW_VAR_NAME_FORMAT_MAPPING = {'AIRFLOW_CONTEXT_DAG_ID': {'default': f'{DEFAULT_FORMAT_PREFIX}dag_id', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}DAG_ID'}, 'AIRFLOW_CONTEXT_TASK_ID': {'default': f'{DEFAULT_FORMAT_PREFIX}task_id', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}TASK_ID'}, 'AIRFLOW_CONTEXT_EXECUTION_DATE': {'default': f'{DEFAULT_FORMAT_PREFIX}execution_date', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}EXECUTION_DATE'}, 'AIRFLOW_CONTEXT_TRY_NUMBER': {'default': f'{DEFAULT_FORMAT_PREFIX}try_number', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}TRY_NUMBER'}, 'AIRFLOW_CONTEXT_DAG_RUN_ID': {'default': f'{DEFAULT_FORMAT_PREFIX}dag_run_id', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}DAG_RUN_ID'}, 'AIRFLOW_CONTEXT_DAG_OWNER': {'default': f'{DEFAULT_FORMAT_PREFIX}dag_owner', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}DAG_OWNER'}, 'AIRFLOW_CONTEXT_DAG_EMAIL': {'default': f'{DEFAULT_FORMAT_PREFIX}dag_email', 'env_var_format': f'{ENV_VAR_FORMAT_PREFIX}DAG_EMAIL'}}

def context_to_airflow_vars(context: Mapping[str, Any], in_env_var_format: bool=False) -> dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return values used to externally reconstruct relations between dags, dag_runs, tasks and task_instances.\n\n    Given a context, this function provides a dictionary of values that can be used to\n    externally reconstruct relations between dags, dag_runs, tasks and task_instances.\n    Default to abc.def.ghi format and can be made to ABC_DEF_GHI format if\n    in_env_var_format is set to True.\n\n    :param context: The context for the task_instance of interest.\n    :param in_env_var_format: If returned vars should be in ABC_DEF_GHI format.\n    :return: task_instance context as dict.\n    '
    params = {}
    if in_env_var_format:
        name_format = 'env_var_format'
    else:
        name_format = 'default'
    task = context.get('task')
    task_instance = context.get('task_instance')
    dag_run = context.get('dag_run')
    ops = [(task, 'email', 'AIRFLOW_CONTEXT_DAG_EMAIL'), (task, 'owner', 'AIRFLOW_CONTEXT_DAG_OWNER'), (task_instance, 'dag_id', 'AIRFLOW_CONTEXT_DAG_ID'), (task_instance, 'task_id', 'AIRFLOW_CONTEXT_TASK_ID'), (task_instance, 'execution_date', 'AIRFLOW_CONTEXT_EXECUTION_DATE'), (task_instance, 'try_number', 'AIRFLOW_CONTEXT_TRY_NUMBER'), (dag_run, 'run_id', 'AIRFLOW_CONTEXT_DAG_RUN_ID')]
    context_params = settings.get_airflow_context_vars(context)
    for (key, value) in context_params.items():
        if not isinstance(key, str):
            raise TypeError(f'key <{key}> must be string')
        if not isinstance(value, str):
            raise TypeError(f'value of key <{key}> must be string, not {type(value)}')
        if in_env_var_format:
            if not key.startswith(ENV_VAR_FORMAT_PREFIX):
                key = ENV_VAR_FORMAT_PREFIX + key.upper()
        elif not key.startswith(DEFAULT_FORMAT_PREFIX):
            key = DEFAULT_FORMAT_PREFIX + key
        params[key] = value
    for (subject, attr, mapping_key) in ops:
        _attr = getattr(subject, attr, None)
        if subject and _attr:
            mapping_value = AIRFLOW_VAR_NAME_FORMAT_MAPPING[mapping_key][name_format]
            if isinstance(_attr, str):
                params[mapping_value] = _attr
            elif isinstance(_attr, datetime):
                params[mapping_value] = _attr.isoformat()
            elif isinstance(_attr, list):
                params[mapping_value] = ','.join(_attr)
            else:
                params[mapping_value] = str(_attr)
    return params

class KeywordParameters:
    """Wrapper representing ``**kwargs`` to a callable.

    The actual ``kwargs`` can be obtained by calling either ``unpacking()`` or
    ``serializing()``. They behave almost the same and are only different if
    the containing ``kwargs`` is an Airflow Context object, and the calling
    function uses ``**kwargs`` in the argument list.

    In this particular case, ``unpacking()`` uses ``lazy-object-proxy`` to
    prevent the Context from emitting deprecation warnings too eagerly when it's
    unpacked by ``**``. ``serializing()`` does not do this, and will allow the
    warnings to be emitted eagerly, which is useful when you want to dump the
    content and use it somewhere else without needing ``lazy-object-proxy``.
    """

    def __init__(self, kwargs: Mapping[str, Any], *, wildcard: bool) -> None:
        if False:
            while True:
                i = 10
        self._kwargs = kwargs
        self._wildcard = wildcard

    @classmethod
    def determine(cls, func: Callable[..., Any], args: Collection[Any], kwargs: Mapping[str, Any]) -> KeywordParameters:
        if False:
            for i in range(10):
                print('nop')
        import inspect
        import itertools
        signature = inspect.signature(func)
        has_wildcard_kwargs = any((p.kind == p.VAR_KEYWORD for p in signature.parameters.values()))
        for name in itertools.islice(signature.parameters.keys(), len(args)):
            if name in kwargs:
                raise ValueError(f'The key {name!r} in args is a part of kwargs and therefore reserved.')
        if has_wildcard_kwargs:
            return cls(kwargs, wildcard=True)
        kwargs = {key: kwargs[key] for key in signature.parameters if key in kwargs}
        return cls(kwargs, wildcard=False)

    def unpacking(self) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        'Dump the kwargs mapping to unpack with ``**`` in a function call.'
        if self._wildcard and isinstance(self._kwargs, Context):
            return lazy_mapping_from_context(self._kwargs)
        return self._kwargs

    def serializing(self) -> Mapping[str, Any]:
        if False:
            return 10
        'Dump the kwargs mapping for serialization purposes.'
        return self._kwargs

def determine_kwargs(func: Callable[..., Any], args: Collection[Any], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    if False:
        return 10
    '\n    Inspect the signature of a callable to determine which kwargs need to be passed to the callable.\n\n    :param func: The callable that you want to invoke\n    :param args: The positional arguments that need to be passed to the callable, so we know how many to skip.\n    :param kwargs: The keyword arguments that need to be filtered before passing to the callable.\n    :return: A dictionary which contains the keyword arguments that are compatible with the callable.\n    '
    return KeywordParameters.determine(func, args, kwargs).unpacking()

def make_kwargs_callable(func: Callable[..., R]) -> Callable[..., R]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a new callable that only forwards necessary arguments from any provided input.\n\n    Make a new callable that can accept any number of positional or keyword arguments\n    but only forwards those required by the given callable func.\n    '
    import functools

    @functools.wraps(func)
    def kwargs_func(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs = determine_kwargs(func, args, kwargs)
        return func(*args, **kwargs)
    return kwargs_func