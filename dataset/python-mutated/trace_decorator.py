import asyncio
from functools import wraps
import inspect
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import cast
from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Tracer
from opentelemetry.trace.span import Span

class TracingDecoratorOptions:

    class NamingSchemes:

        @staticmethod
        def function_qualified_name(func: Callable) -> str:
            if False:
                print('Hello World!')
            return func.__qualname__
        default_scheme = function_qualified_name
    naming_scheme: ClassVar[Callable[[Callable], str]] = NamingSchemes.default_scheme
    default_attributes: ClassVar[Dict[str, str]] = {}

    @classmethod
    def set_naming_scheme(cls, naming_scheme: Callable[[Callable], str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls.naming_scheme = naming_scheme

    @classmethod
    def set_default_attributes(cls, attributes: Optional[Dict[str, str]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if attributes is not None:
            for att in attributes:
                cls.default_attributes[att] = attributes[att]
T = TypeVar('T', bound=Union[Callable, type])

def instrument(_func_or_class: T, *, span_name: str='', record_exception: bool=True, attributes: Optional[Dict[str, str]]=None, existing_tracer: Optional[Tracer]=None, ignore: bool=False) -> T:
    if False:
        print('Hello World!')
    '\n    A decorator to instrument a class or function with an OTEL tracing span.\n    :param cls: internal, used to specify scope of instrumentation\n    :param _func_or_class: The function or span to instrument, this is automatically assigned\n    :param span_name: Specify the span name explicitly, rather than use the naming convention.\n    This parameter has no effect for class decorators: str\n    :param record_exception: Sets whether any exceptions occurring in the span and the stacktrace are recorded\n    automatically: bool\n    :param attributes:A dictionary of span attributes. These will be automatically added to the span. If defined on a\n    class decorator, they will be added to every function span under the class.: dict\n    :param existing_tracer: Use a specific tracer instead of creating one :Tracer\n    :param ignore: Do not instrument this function, has no effect for class decorators:bool\n    :return:The decorator function\n    '

    def decorate_class(cls: T) -> T:
        if False:
            return 10
        for (name, method) in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_'):
                if isinstance(inspect.getattr_static(cls, name), staticmethod):
                    setattr(cls, name, staticmethod(instrument(method, record_exception=record_exception, attributes=attributes, existing_tracer=existing_tracer)))
                else:
                    setattr(cls, name, instrument(method, record_exception=record_exception, attributes=attributes, existing_tracer=existing_tracer))
        return cls
    if inspect.isclass(_func_or_class):
        return decorate_class(_func_or_class)

    def span_decorator(func_or_class: T) -> T:
        if False:
            i = 10
            return i + 15
        if inspect.isclass(func_or_class):
            return decorate_class(func_or_class)
        undecorated_func = getattr(func_or_class, '__tracing_unwrapped__', None)
        if undecorated_func:
            return func_or_class
        func_or_class.__tracing_unwrapped__ = func_or_class
        tracer = existing_tracer or trace.get_tracer(func_or_class.__module__)

        def _set_semantic_attributes(span: Span, func: Callable) -> None:
            if False:
                for i in range(10):
                    print('nop')
            span.set_attribute(SpanAttributes.CODE_NAMESPACE, func.__module__)
            span.set_attribute(SpanAttributes.CODE_FUNCTION, func.__qualname__)
            span.set_attribute(SpanAttributes.CODE_FILEPATH, func.__code__.co_filename)
            span.set_attribute(SpanAttributes.CODE_LINENO, func.__code__.co_firstlineno)

        def _set_attributes(span: Span, attributes_dict: Optional[Dict[str, str]]=None) -> None:
            if False:
                for i in range(10):
                    print('nop')
            if attributes_dict is not None:
                for att in attributes_dict:
                    span.set_attribute(att, attributes_dict[att])

        @wraps(func_or_class)
        def wrap_with_span_sync(*args, **kwargs):
            if False:
                print('Hello World!')
            name = span_name or TracingDecoratorOptions.naming_scheme(func_or_class)
            with tracer.start_as_current_span(name, record_exception=record_exception) as span:
                _set_semantic_attributes(span, func_or_class)
                _set_attributes(span, TracingDecoratorOptions.default_attributes)
                _set_attributes(span, attributes)
                return func_or_class(*args, **kwargs)

        @wraps(func_or_class)
        async def wrap_with_span_async(*args, **kwargs):
            name = span_name or TracingDecoratorOptions.naming_scheme(func_or_class)
            with tracer.start_as_current_span(name, record_exception=record_exception) as span:
                _set_semantic_attributes(span, func_or_class)
                _set_attributes(span, TracingDecoratorOptions.default_attributes)
                _set_attributes(span, attributes)
                return await func_or_class(*args, **kwargs)
        if ignore:
            return func_or_class
        wrapper = wrap_with_span_async if asyncio.iscoroutinefunction(func_or_class) else wrap_with_span_sync
        wrapper.__signature__ = inspect.signature(func_or_class)
        return cast(T, wrapper)
    return span_decorator(_func_or_class)