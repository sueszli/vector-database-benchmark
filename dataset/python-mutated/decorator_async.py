"""The decorator to apply if you want the given function traced."""
import functools
from typing import Awaitable, Callable, Any, TypeVar, overload, Optional
from typing_extensions import ParamSpec
from .common import change_context, get_function_and_class_name
from . import SpanKind as _SpanKind
from ..settings import settings
P = ParamSpec('P')
T = TypeVar('T')

@overload
def distributed_trace_async(__func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    if False:
        for i in range(10):
            print('nop')
    pass

@overload
def distributed_trace_async(**kwargs: Any) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    if False:
        while True:
            i = 10
    pass

def distributed_trace_async(__func: Optional[Callable[P, Awaitable[T]]]=None, **kwargs: Any) -> Any:
    if False:
        while True:
            i = 10
    'Decorator to apply to function to get traced automatically.\n\n    Span will use the func name or "name_of_span".\n\n    :param callable func: A function to decorate\n    :keyword name_of_span: The span name to replace func name if necessary\n    :paramtype name_of_span: str\n    :keyword kind: The kind of the span. INTERNAL by default.\n    :paramtype kind: ~azure.core.tracing.SpanKind\n    '
    name_of_span = kwargs.pop('name_of_span', None)
    tracing_attributes = kwargs.pop('tracing_attributes', {})
    kind = kwargs.pop('kind', _SpanKind.INTERNAL)

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        if False:
            print('Hello World!')

        @functools.wraps(func)
        async def wrapper_use_tracer(*args: Any, **kwargs: Any) -> T:
            merge_span = kwargs.pop('merge_span', False)
            passed_in_parent = kwargs.pop('parent_span', None)
            span_impl_type = settings.tracing_implementation()
            if span_impl_type is None:
                return await func(*args, **kwargs)
            if merge_span and (not passed_in_parent):
                return await func(*args, **kwargs)
            with change_context(passed_in_parent):
                name = name_of_span or get_function_and_class_name(func, *args)
                with span_impl_type(name=name, kind=kind) as span:
                    for (key, value) in tracing_attributes.items():
                        span.add_attribute(key, value)
                    return await func(*args, **kwargs)
        return wrapper_use_tracer
    return decorator if __func is None else decorator(__func)