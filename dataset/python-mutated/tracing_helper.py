import importlib
import inspect
import logging
import os
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter
from types import ModuleType
from typing import Any, Callable, Dict, Generator, List, MutableMapping, Optional, Sequence, Union, cast
import ray
import ray._private.worker
from ray._private.inspect_util import is_class_method, is_function_or_method, is_static_method
from ray.runtime_context import get_runtime_context
logger = logging.getLogger(__name__)

class _OpenTelemetryProxy:
    """
    This proxy makes it possible for tracing to be disabled when opentelemetry
    is not installed on the cluster, but is installed locally.

    The check for `opentelemetry`'s existence must happen where the functions
    are executed because `opentelemetry` may be present where the functions
    are pickled. This can happen when `ray[full]` is installed locally by `ray`
    (no extra dependencies) is installed on the cluster.
    """
    allowed_functions = {'trace', 'context', 'propagate', 'Context'}

    def __getattr__(self, name):
        if False:
            return 10
        if name in _OpenTelemetryProxy.allowed_functions:
            return getattr(self, f'_{name}')()
        else:
            raise AttributeError(f'Attribute does not exist: {name}')

    def _trace(self):
        if False:
            print('Hello World!')
        return self._try_import('opentelemetry.trace')

    def _context(self):
        if False:
            print('Hello World!')
        return self._try_import('opentelemetry.context')

    def _propagate(self):
        if False:
            i = 10
            return i + 15
        return self._try_import('opentelemetry.propagate')

    def _Context(self):
        if False:
            print('Hello World!')
        context = self._context()
        if context:
            return context.context.Context
        else:
            return None

    def try_all(self):
        if False:
            return 10
        self._trace()
        self._context()
        self._propagate()
        self._Context()

    def _try_import(self, module):
        if False:
            i = 10
            return i + 15
        try:
            return importlib.import_module(module)
        except ImportError:
            if os.getenv('RAY_TRACING_ENABLED', 'False').lower() in ['true', '1']:
                raise ImportError("Install opentelemetry with 'pip install opentelemetry-api==1.0.0rc1' and 'pip install opentelemetry-sdk==1.0.0rc1' to enable tracing. See more at docs.ray.io/tracing.html")
_global_is_tracing_enabled = False
_opentelemetry = None

def _is_tracing_enabled() -> bool:
    if False:
        i = 10
        return i + 15
    'Checks environment variable feature flag to see if tracing is turned on.\n    Tracing is off by default.'
    return _global_is_tracing_enabled

def _enable_tracing():
    if False:
        return 10
    global _global_is_tracing_enabled, _opentelemetry
    _global_is_tracing_enabled = True
    _opentelemetry = _OpenTelemetryProxy()
    _opentelemetry.try_all()

def _sort_params_list(params_list: List[Parameter]):
    if False:
        print('Hello World!')
    'Given a list of Parameters, if a kwargs Parameter exists,\n    move it to the end of the list.'
    for (i, param) in enumerate(params_list):
        if param.kind == Parameter.VAR_KEYWORD:
            params_list.append(params_list.pop(i))
            break
    return params_list

def _add_param_to_signature(function: Callable, new_param: Parameter):
    if False:
        return 10
    'Add additional Parameter to function signature.'
    old_sig = inspect.signature(function)
    old_sig_list_repr = list(old_sig.parameters.values())
    if any((param.name == new_param.name for param in old_sig_list_repr)):
        return old_sig
    new_params = _sort_params_list(old_sig_list_repr + [new_param])
    new_sig = old_sig.replace(parameters=new_params)
    return new_sig

class _ImportFromStringError(Exception):
    pass

def _import_from_string(import_str: Union[ModuleType, str]) -> ModuleType:
    if False:
        while True:
            i = 10
    'Given a string that is in format "<module>:<attribute>",\n    import the attribute.'
    if not isinstance(import_str, str):
        return import_str
    (module_str, _, attrs_str) = import_str.partition(':')
    if not module_str or not attrs_str:
        message = 'Import string "{import_str}" must be in format"<module>:<attribute>".'
        raise _ImportFromStringError(message.format(import_str=import_str))
    try:
        module = importlib.import_module(module_str)
    except ImportError as exc:
        if exc.name != module_str:
            raise exc from None
        message = 'Could not import module "{module_str}".'
        raise _ImportFromStringError(message.format(module_str=module_str))
    instance = module
    try:
        for attr_str in attrs_str.split('.'):
            instance = getattr(instance, attr_str)
    except AttributeError:
        message = 'Attribute "{attrs_str}" not found in module "{module_str}".'
        raise _ImportFromStringError(message.format(attrs_str=attrs_str, module_str=module_str))
    return instance

class _DictPropagator:

    def inject_current_context() -> Dict[Any, Any]:
        if False:
            while True:
                i = 10
        'Inject trace context into otel propagator.'
        context_dict: Dict[Any, Any] = {}
        _opentelemetry.propagate.inject(context_dict)
        return context_dict

    def extract(context_dict: Dict[Any, Any]) -> '_opentelemetry.Context':
        if False:
            while True:
                i = 10
        'Given a trace context, extract as a Context.'
        return cast(_opentelemetry.Context, _opentelemetry.propagate.extract(context_dict))

@contextmanager
def _use_context(parent_context: '_opentelemetry.Context') -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Ray trace context for the span.'
    if parent_context is not None:
        new_context = parent_context
    else:
        new_context = _opentelemetry.Context()
    token = _opentelemetry.context.attach(new_context)
    try:
        yield
    finally:
        _opentelemetry.context.detach(token)

def _function_hydrate_span_args(function_name: str):
    if False:
        while True:
            i = 10
    'Get the Attributes of the function that will be reported as attributes\n    in the trace.'
    runtime_context = get_runtime_context()
    span_args = {'ray.remote': 'function', 'ray.function': function_name, 'ray.pid': str(os.getpid()), 'ray.job_id': runtime_context.get_job_id(), 'ray.node_id': runtime_context.get_node_id()}
    if ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE:
        task_id = runtime_context.get_task_id()
        if task_id:
            span_args['ray.task_id'] = task_id
    worker_id = getattr(ray._private.worker.global_worker, 'worker_id', None)
    if worker_id:
        span_args['ray.worker_id'] = worker_id.hex()
    return span_args

def _function_span_producer_name(func: Callable[..., Any]) -> str:
    if False:
        i = 10
        return i + 15
    'Returns the function span name that has span kind of producer.'
    return f'{func} ray.remote'

def _function_span_consumer_name(func: Callable[..., Any]) -> str:
    if False:
        i = 10
        return i + 15
    'Returns the function span name that has span kind of consumer.'
    return f'{func} ray.remote_worker'

def _actor_hydrate_span_args(class_: Union[str, Callable[..., Any]], method: Union[str, Callable[..., Any]]):
    if False:
        print('Hello World!')
    'Get the Attributes of the actor that will be reported as attributes\n    in the trace.'
    if callable(class_):
        class_ = class_.__name__
    if callable(method):
        method = method.__name__
    runtime_context = get_runtime_context()
    span_args = {'ray.remote': 'actor', 'ray.actor_class': class_, 'ray.actor_method': method, 'ray.function': f'{class_}.{method}', 'ray.pid': str(os.getpid()), 'ray.job_id': runtime_context.get_job_id(), 'ray.node_id': runtime_context.get_node_id()}
    if ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE:
        actor_id = runtime_context.get_actor_id()
        if actor_id:
            span_args['ray.actor_id'] = actor_id
    worker_id = getattr(ray._private.worker.global_worker, 'worker_id', None)
    if worker_id:
        span_args['ray.worker_id'] = worker_id.hex()
    return span_args

def _actor_span_producer_name(class_: Union[str, Callable[..., Any]], method: Union[str, Callable[..., Any]]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns the actor span name that has span kind of producer.'
    if not isinstance(class_, str):
        class_ = class_.__name__
    if not isinstance(method, str):
        method = method.__name__
    return f'{class_}.{method} ray.remote'

def _actor_span_consumer_name(class_: Union[str, Callable[..., Any]], method: Union[str, Callable[..., Any]]) -> str:
    if False:
        while True:
            i = 10
    'Returns the actor span name that has span kind of consumer.'
    if not isinstance(class_, str):
        class_ = class_.__name__
    if not isinstance(method, str):
        method = method.__name__
    return f'{class_}.{method} ray.remote_worker'

def _tracing_task_invocation(method):
    if False:
        for i in range(10):
            print('nop')
    'Trace the execution of a remote task. Inject\n    the current span context into kwargs for propagation.'

    @wraps(method)
    def _invocation_remote_span(self, args: Any=None, kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if not _is_tracing_enabled() or self._is_cross_language:
            if kwargs is not None:
                assert '_ray_trace_ctx' not in kwargs
            return method(self, args, kwargs, *_args, **_kwargs)
        assert '_ray_trace_ctx' not in kwargs
        tracer = _opentelemetry.trace.get_tracer(__name__)
        with tracer.start_as_current_span(_function_span_producer_name(self._function_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_function_hydrate_span_args(self._function_name)):
            kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
            return method(self, args, kwargs, *_args, **_kwargs)
    return _invocation_remote_span

def _inject_tracing_into_function(function):
    if False:
        print('Hello World!')
    "Wrap the function argument passed to RemoteFunction's __init__ so that\n    future execution of that function will include tracing.\n    Use the provided trace context from kwargs.\n    "
    if not _is_tracing_enabled():
        return function
    setattr(function, '__signature__', _add_param_to_signature(function, inspect.Parameter('_ray_trace_ctx', inspect.Parameter.KEYWORD_ONLY, default=None)))

    @wraps(function)
    def _function_with_tracing(*args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
        if False:
            while True:
                i = 10
        if _ray_trace_ctx is None:
            return function(*args, **kwargs)
        tracer = _opentelemetry.trace.get_tracer(__name__)
        function_name = function.__module__ + '.' + function.__name__
        with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_function_span_consumer_name(function_name), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_function_hydrate_span_args(function_name)):
            return function(*args, **kwargs)
    return _function_with_tracing

def _tracing_actor_creation(method):
    if False:
        return 10
    'Trace the creation of an actor. Inject\n    the current span context into kwargs for propagation.'

    @wraps(method)
    def _invocation_actor_class_remote_span(self, args: Any=tuple(), kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any):
        if False:
            while True:
                i = 10
        if kwargs is None:
            kwargs = {}
        if not _is_tracing_enabled():
            assert '_ray_trace_ctx' not in kwargs
            return method(self, args, kwargs, *_args, **_kwargs)
        class_name = self.__ray_metadata__.class_name
        method_name = '__init__'
        assert '_ray_trace_ctx' not in _kwargs
        tracer = _opentelemetry.trace.get_tracer(__name__)
        with tracer.start_as_current_span(name=_actor_span_producer_name(class_name, method_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_actor_hydrate_span_args(class_name, method_name)) as span:
            kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
            result = method(self, args, kwargs, *_args, **_kwargs)
            span.set_attribute('ray.actor_id', result._ray_actor_id.hex())
            return result
    return _invocation_actor_class_remote_span

def _tracing_actor_method_invocation(method):
    if False:
        i = 10
        return i + 15
    'Trace the invocation of an actor method.'

    @wraps(method)
    def _start_span(self, args: Sequence[Any]=None, kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        if not _is_tracing_enabled() or self._actor_ref()._ray_is_cross_language:
            if kwargs is not None:
                assert '_ray_trace_ctx' not in kwargs
            return method(self, args, kwargs, *_args, **_kwargs)
        class_name = self._actor_ref()._ray_actor_creation_function_descriptor.class_name
        method_name = self._method_name
        assert '_ray_trace_ctx' not in _kwargs
        tracer = _opentelemetry.trace.get_tracer(__name__)
        with tracer.start_as_current_span(name=_actor_span_producer_name(class_name, method_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_actor_hydrate_span_args(class_name, method_name)) as span:
            kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
            span.set_attribute('ray.actor_id', self._actor_ref()._ray_actor_id.hex())
            return method(self, args, kwargs, *_args, **_kwargs)
    return _start_span

def _inject_tracing_into_class(_cls):
    if False:
        i = 10
        return i + 15
    'Given a class that will be made into an actor,\n    inject tracing into all of the methods.'

    def span_wrapper(method: Callable[..., Any]) -> Any:
        if False:
            i = 10
            return i + 15

        def _resume_span(self: Any, *_args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **_kwargs: Any) -> Any:
            if False:
                i = 10
                return i + 15
            "\n            Wrap the user's function with a function that\n            will extract the trace context\n            "
            if not _is_tracing_enabled() or _ray_trace_ctx is None:
                return method(self, *_args, **_kwargs)
            tracer: _opentelemetry.trace.Tracer = _opentelemetry.trace.get_tracer(__name__)
            with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_actor_span_consumer_name(self.__class__.__name__, method), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_actor_hydrate_span_args(self.__class__.__name__, method)):
                return method(self, *_args, **_kwargs)
        return _resume_span

    def async_span_wrapper(method: Callable[..., Any]) -> Any:
        if False:
            print('Hello World!')

        async def _resume_span(self: Any, *_args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **_kwargs: Any) -> Any:
            """
            Wrap the user's function with a function that
            will extract the trace context
            """
            if not _is_tracing_enabled() or _ray_trace_ctx is None:
                return await method(self, *_args, **_kwargs)
            tracer = _opentelemetry.trace.get_tracer(__name__)
            with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_actor_span_consumer_name(self.__class__.__name__, method.__name__), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_actor_hydrate_span_args(self.__class__.__name__, method.__name__)):
                return await method(self, *_args, **_kwargs)
        return _resume_span
    methods = inspect.getmembers(_cls, is_function_or_method)
    for (name, method) in methods:
        if is_static_method(_cls, name) or is_class_method(method):
            continue
        if inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method):
            continue
        if name == '__del__':
            continue
        setattr(method, '__signature__', _add_param_to_signature(method, inspect.Parameter('_ray_trace_ctx', inspect.Parameter.KEYWORD_ONLY, default=None)))
        if inspect.iscoroutinefunction(method):
            wrapped_method = wraps(method)(async_span_wrapper(method))
        else:
            wrapped_method = wraps(method)(span_wrapper(method))
        setattr(_cls, name, wrapped_method)
    return _cls