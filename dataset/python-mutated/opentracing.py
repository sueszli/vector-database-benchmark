"""
============================
Using OpenTracing in Synapse
============================

Python-specific tracing concepts are at https://opentracing.io/guides/python/.
Note that Synapse wraps OpenTracing in a small module (this one) in order to make the
OpenTracing dependency optional. That means that the access patterns are
different to those demonstrated in the OpenTracing guides. However, it is
still useful to know, especially if OpenTracing is included as a full dependency
in the future or if you are modifying this module.


OpenTracing is encapsulated so that
no span objects from OpenTracing are exposed in Synapse's code. This allows
OpenTracing to be easily disabled in Synapse and thereby have OpenTracing as
an optional dependency. This does however limit the number of modifiable spans
at any point in the code to one. From here out references to `opentracing`
in the code snippets refer to the Synapses module.
Most methods provided in the module have a direct correlation to those provided
by opentracing. Refer to docs there for a more in-depth documentation on some of
the args and methods.

Tracing
-------

In Synapse it is not possible to start a non-active span. Spans can be started
using the ``start_active_span`` method. This returns a scope (see
OpenTracing docs) which is a context manager that needs to be entered and
exited. This is usually done by using ``with``.

.. code-block:: python

   from synapse.logging.opentracing import start_active_span

   with start_active_span("operation name"):
       # Do something we want to tracer

Forgetting to enter or exit a scope will result in some mysterious and grievous log
context errors.

At anytime where there is an active span ``opentracing.set_tag`` can be used to
set a tag on the current active span.

Tracing functions
-----------------

Functions can be easily traced using decorators. The name of
the function becomes the operation name for the span.

.. code-block:: python

   from synapse.logging.opentracing import trace

   # Start a span using 'interesting_function' as the operation name
   @trace
   def interesting_function(*args, **kwargs):
       # Does all kinds of cool and expected things
       return something_usual_and_useful


Operation names can be explicitly set for a function by using ``trace_with_opname``:

.. code-block:: python

   from synapse.logging.opentracing import trace_with_opname

   @trace_with_opname("a_better_operation_name")
   def interesting_badly_named_function(*args, **kwargs):
       # Does all kinds of cool and expected things
       return something_usual_and_useful

Setting Tags
------------

To set a tag on the active span do

.. code-block:: python

   from synapse.logging.opentracing import set_tag

   set_tag(tag_name, tag_value)

There's a convenient decorator to tag all the args of the method. It uses
inspection in order to use the formal parameter names prefixed with 'ARG_' as
tag names. It uses kwarg names as tag names without the prefix.

.. code-block:: python

   from synapse.logging.opentracing import tag_args

   @tag_args
   def set_fates(clotho, lachesis, atropos, father="Zues", mother="Themis"):
       pass

   set_fates("the story", "the end", "the act")
   # This will have the following tags
   #  - ARG_clotho: "the story"
   #  - ARG_lachesis: "the end"
   #  - ARG_atropos: "the act"
   #  - father: "Zues"
   #  - mother: "Themis"

Contexts and carriers
---------------------

There are a selection of wrappers for injecting and extracting contexts from
carriers provided. Unfortunately OpenTracing's three context injection
techniques are not adequate for our inject of OpenTracing span-contexts into
Twisted's http headers, EDU contents and our database tables. Also note that
the binary encoding format mandated by OpenTracing is not actually implemented
by jaeger_client v4.0.0 - it will silently noop.
Please refer to the end of ``logging/opentracing.py`` for the available
injection and extraction methods.

Homeserver whitelisting
-----------------------

Most of the whitelist checks are encapsulated in the modules's injection
and extraction method but be aware that using custom carriers or crossing
unchartered waters will require the enforcement of the whitelist.
``logging/opentracing.py`` has a ``whitelisted_homeserver`` method which takes
in a destination and compares it to the whitelist.

Most injection methods take a 'destination' arg. The context will only be injected
if the destination matches the whitelist or the destination is None.

=======
Gotchas
=======

- Checking whitelists on span propagation
- Inserting pii
- Forgetting to enter or exit a scope
- Span source: make sure that the span you expect to be active across a
  function call really will be that one. Does the current function have more
  than one caller? Will all of those calling functions have be in a context
  with an active span?
"""
import contextlib
import enum
import inspect
import logging
import re
from functools import wraps
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Collection, ContextManager, Dict, Generator, Iterable, List, Optional, Pattern, Type, TypeVar, Union, cast, overload
import attr
from typing_extensions import Concatenate, ParamSpec
from twisted.internet import defer
from twisted.web.http import Request
from twisted.web.http_headers import Headers
from synapse.config import ConfigError
from synapse.util import json_decoder, json_encoder
if TYPE_CHECKING:
    from synapse.http.site import SynapseRequest
    from synapse.server import HomeServer
STRIP_INSTANCE_NUMBER_SUFFIX_REGEX = re.compile('[_-]?\\d+$')

class _DummyTagNames:
    """wrapper of opentracings tags. We need to have them if we
    want to reference them without opentracing around. Clearly they
    should never actually show up in a trace. `set_tags` overwrites
    these with the correct ones."""
    INVALID_TAG = 'invalid-tag'
    COMPONENT = INVALID_TAG
    DATABASE_INSTANCE = INVALID_TAG
    DATABASE_STATEMENT = INVALID_TAG
    DATABASE_TYPE = INVALID_TAG
    DATABASE_USER = INVALID_TAG
    ERROR = INVALID_TAG
    HTTP_METHOD = INVALID_TAG
    HTTP_STATUS_CODE = INVALID_TAG
    HTTP_URL = INVALID_TAG
    MESSAGE_BUS_DESTINATION = INVALID_TAG
    PEER_ADDRESS = INVALID_TAG
    PEER_HOSTNAME = INVALID_TAG
    PEER_HOST_IPV4 = INVALID_TAG
    PEER_HOST_IPV6 = INVALID_TAG
    PEER_PORT = INVALID_TAG
    PEER_SERVICE = INVALID_TAG
    SAMPLING_PRIORITY = INVALID_TAG
    SERVICE = INVALID_TAG
    SPAN_KIND = INVALID_TAG
    SPAN_KIND_CONSUMER = INVALID_TAG
    SPAN_KIND_PRODUCER = INVALID_TAG
    SPAN_KIND_RPC_CLIENT = INVALID_TAG
    SPAN_KIND_RPC_SERVER = INVALID_TAG
try:
    import opentracing
    import opentracing.tags
    tags = opentracing.tags
except ImportError:
    opentracing = None
    tags = _DummyTagNames
try:
    from jaeger_client import Config as JaegerConfig
    from synapse.logging.scopecontextmanager import LogContextScopeManager
except ImportError:
    JaegerConfig = None
    LogContextScopeManager = None
try:
    from rust_python_jaeger_reporter import Reporter
    try:
        from jaeger_client.reporter import BaseReporter
    except ImportError:

        class BaseReporter:
            pass

    @attr.s(slots=True, frozen=True, auto_attribs=True)
    class _WrappedRustReporter(BaseReporter):
        """Wrap the reporter to ensure `report_span` never throws."""
        _reporter: Reporter = attr.Factory(Reporter)

        def set_process(self, *args: Any, **kwargs: Any) -> None:
            if False:
                print('Hello World!')
            return self._reporter.set_process(*args, **kwargs)

        def report_span(self, span: 'opentracing.Span') -> None:
            if False:
                print('Hello World!')
            try:
                return self._reporter.report_span(span)
            except Exception:
                logger.exception('Failed to report span')
    RustReporter: Optional[Type[_WrappedRustReporter]] = _WrappedRustReporter
except ImportError:
    RustReporter = None
logger = logging.getLogger(__name__)

class SynapseTags:
    TO_DEVICE_EDU_ID = 'to_device.edu_id'
    TO_DEVICE_TYPE = 'to_device.type'
    TO_DEVICE_SENDER = 'to_device.sender'
    TO_DEVICE_RECIPIENT = 'to_device.recipient'
    TO_DEVICE_RECIPIENT_DEVICE = 'to_device.recipient_device'
    TO_DEVICE_MSGID = 'to_device.msgid'
    SYNC_RESULT = 'sync.new_data'
    INSTANCE_NAME = 'instance_name'
    REQUEST_ID = 'request_id'
    REQUEST_TAG = 'request_tag'
    DB_TXN_DESC = 'db.txn_desc'
    DB_TXN_ID = 'db.txn_id'
    CACHE_NAME = 'cache.name'
    SEND_JOIN_RESPONSE_IS_PARTIAL_STATE = 'send_join.partial_state_response'
    FUNC_ARG_PREFIX = 'ARG.'
    FUNC_ARGS = 'args'
    FUNC_KWARGS = 'kwargs'
    RESULT_PREFIX = 'RESULT.'

class SynapseBaggage:
    FORCE_TRACING = 'synapse-force-tracing'
_homeserver_whitelist: Optional[Pattern[str]] = None

class _Sentinel(enum.Enum):
    sentinel = object()
P = ParamSpec('P')
R = TypeVar('R')
T = TypeVar('T')

def only_if_tracing(func: Callable[P, R]) -> Callable[P, Optional[R]]:
    if False:
        i = 10
        return i + 15
    "Executes the function only if we're tracing. Otherwise returns None."

    @wraps(func)
    def _only_if_tracing_inner(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
        if False:
            print('Hello World!')
        if opentracing:
            return func(*args, **kwargs)
        else:
            return None
    return _only_if_tracing_inner

@overload
def ensure_active_span(message: str) -> Callable[[Callable[P, R]], Callable[P, Optional[R]]]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def ensure_active_span(message: str, ret: T) -> Callable[[Callable[P, R]], Callable[P, Union[T, R]]]:
    if False:
        while True:
            i = 10
    ...

def ensure_active_span(message: str, ret: Optional[T]=None) -> Callable[[Callable[P, R]], Callable[P, Union[Optional[T], R]]]:
    if False:
        i = 10
        return i + 15
    'Executes the operation only if opentracing is enabled and there is an active span.\n    If there is no active span it logs message at the error level.\n\n    Args:\n        message: Message which fills in "There was no active span when trying to %s"\n            in the error log if there is no active span and opentracing is enabled.\n        ret: return value if opentracing is None or there is no active span.\n\n    Returns:\n        The result of the func, falling back to ret if opentracing is disabled or there\n        was no active span.\n    '

    def ensure_active_span_inner_1(func: Callable[P, R]) -> Callable[P, Union[Optional[T], R]]:
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def ensure_active_span_inner_2(*args: P.args, **kwargs: P.kwargs) -> Union[Optional[T], R]:
            if False:
                for i in range(10):
                    print('nop')
            if not opentracing:
                return ret
            if not opentracing.tracer.active_span:
                logger.error('There was no active span when trying to %s. Did you forget to start one or did a context slip?', message, stack_info=True)
                return ret
            return func(*args, **kwargs)
        return ensure_active_span_inner_2
    return ensure_active_span_inner_1

def init_tracer(hs: 'HomeServer') -> None:
    if False:
        while True:
            i = 10
    'Set the whitelists and initialise the JaegerClient tracer'
    global opentracing
    if not hs.config.tracing.opentracer_enabled:
        opentracing = None
        return
    if opentracing is None or JaegerConfig is None:
        raise ConfigError('The server has been configured to use opentracing but opentracing is not installed.')
    set_homeserver_whitelist(hs.config.tracing.opentracer_whitelist)
    from jaeger_client.metrics.prometheus import PrometheusMetricsFactory
    instance_name_by_type = re.sub(STRIP_INSTANCE_NUMBER_SUFFIX_REGEX, '', hs.get_instance_name())
    jaeger_config = hs.config.tracing.jaeger_config
    tags = jaeger_config.setdefault('tags', {})
    tags[SynapseTags.INSTANCE_NAME] = hs.get_instance_name()
    config = JaegerConfig(config=jaeger_config, service_name=f'{hs.config.server.server_name} {instance_name_by_type}', scope_manager=LogContextScopeManager(), metrics_factory=PrometheusMetricsFactory())
    if RustReporter:
        logger.info('Using rust_python_jaeger_reporter library')
        assert config.sampler is not None
        tracer = config.create_tracer(RustReporter(), config.sampler)
        opentracing.set_global_tracer(tracer)
    else:
        config.initialize_tracer()

@only_if_tracing
def set_homeserver_whitelist(homeserver_whitelist: Iterable[str]) -> None:
    if False:
        return 10
    'Sets the homeserver whitelist\n\n    Args:\n        homeserver_whitelist: regexes specifying whitelisted homeservers\n    '
    global _homeserver_whitelist
    if homeserver_whitelist:
        _homeserver_whitelist = re.compile('({})'.format(')|('.join(homeserver_whitelist)))

@only_if_tracing
def whitelisted_homeserver(destination: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if a destination matches the whitelist\n\n    Args:\n        destination\n    '
    if _homeserver_whitelist:
        return _homeserver_whitelist.match(destination) is not None
    return False

def start_active_span(operation_name: str, child_of: Optional[Union['opentracing.Span', 'opentracing.SpanContext']]=None, references: Optional[List['opentracing.Reference']]=None, tags: Optional[Dict[str, str]]=None, start_time: Optional[float]=None, ignore_active_span: bool=False, finish_on_close: bool=True, *, tracer: Optional['opentracing.Tracer']=None) -> 'opentracing.Scope':
    if False:
        while True:
            i = 10
    'Starts an active opentracing span.\n\n    Records the start time for the span, and sets it as the "active span" in the\n    scope manager.\n\n    Args:\n        See opentracing.tracer\n    Returns:\n        scope (Scope) or contextlib.nullcontext\n    '
    if opentracing is None:
        return contextlib.nullcontext()
    if tracer is None:
        tracer = opentracing.tracer
    return tracer.start_active_span(operation_name, child_of=child_of, references=references, tags=tags, start_time=start_time, ignore_active_span=ignore_active_span, finish_on_close=finish_on_close)

def start_active_span_follows_from(operation_name: str, contexts: Collection, child_of: Optional[Union['opentracing.Span', 'opentracing.SpanContext']]=None, start_time: Optional[float]=None, *, inherit_force_tracing: bool=False, tracer: Optional['opentracing.Tracer']=None) -> 'opentracing.Scope':
    if False:
        while True:
            i = 10
    'Starts an active opentracing span, with additional references to previous spans\n\n    Args:\n        operation_name: name of the operation represented by the new span\n        contexts: the previous spans to inherit from\n\n        child_of: optionally override the parent span. If unset, the currently active\n           span will be the parent. (If there is no currently active span, the first\n           span in `contexts` will be the parent.)\n\n        start_time: optional override for the start time of the created span. Seconds\n            since the epoch.\n\n        inherit_force_tracing: if set, and any of the previous contexts have had tracing\n           forced, the new span will also have tracing forced.\n        tracer: override the opentracing tracer. By default the global tracer is used.\n    '
    if opentracing is None:
        return contextlib.nullcontext()
    references = [opentracing.follows_from(context) for context in contexts]
    scope = start_active_span(operation_name, child_of=child_of, references=references, start_time=start_time, tracer=tracer)
    if inherit_force_tracing and any((is_context_forced_tracing(ctx) for ctx in contexts)):
        force_tracing(scope.span)
    return scope

def start_active_span_from_edu(edu_content: Dict[str, Any], operation_name: str, references: Optional[List['opentracing.Reference']]=None, tags: Optional[Dict[str, str]]=None, start_time: Optional[float]=None, ignore_active_span: bool=False, finish_on_close: bool=True) -> 'opentracing.Scope':
    if False:
        while True:
            i = 10
    '\n    Extracts a span context from an edu and uses it to start a new active span\n\n    Args:\n        edu_content: an edu_content with a `context` field whose value is\n        canonical json for a dict which contains opentracing information.\n\n        For the other args see opentracing.tracer\n    '
    references = references or []
    if opentracing is None:
        return contextlib.nullcontext()
    carrier = json_decoder.decode(edu_content.get('context', '{}')).get('opentracing', {})
    context = opentracing.tracer.extract(opentracing.Format.TEXT_MAP, carrier)
    _references = [opentracing.child_of(span_context_from_string(x)) for x in carrier.get('references', [])]
    references += _references
    scope = opentracing.tracer.start_active_span(operation_name, child_of=context, references=references, tags=tags, start_time=start_time, ignore_active_span=ignore_active_span, finish_on_close=finish_on_close)
    scope.span.set_tag('references', carrier.get('references', []))
    return scope

@only_if_tracing
def active_span() -> Optional['opentracing.Span']:
    if False:
        return 10
    'Get the currently active span, if any'
    return opentracing.tracer.active_span

@ensure_active_span('set a tag')
def set_tag(key: str, value: Union[str, bool, int, float]) -> None:
    if False:
        return 10
    'Sets a tag on the active span'
    assert opentracing.tracer.active_span is not None
    opentracing.tracer.active_span.set_tag(key, value)

@ensure_active_span('log')
def log_kv(key_values: Dict[str, Any], timestamp: Optional[float]=None) -> None:
    if False:
        return 10
    'Log to the active span'
    assert opentracing.tracer.active_span is not None
    opentracing.tracer.active_span.log_kv(key_values, timestamp)

@ensure_active_span('set the traces operation name')
def set_operation_name(operation_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Sets the operation name of the active span'
    assert opentracing.tracer.active_span is not None
    opentracing.tracer.active_span.set_operation_name(operation_name)

@only_if_tracing
def force_tracing(span: Union['opentracing.Span', _Sentinel]=_Sentinel.sentinel) -> None:
    if False:
        return 10
    'Force sampling for the active/given span and its children.\n\n    Args:\n        span: span to force tracing for. By default, the active span.\n    '
    if isinstance(span, _Sentinel):
        span_to_trace = opentracing.tracer.active_span
    else:
        span_to_trace = span
    if span_to_trace is None:
        logger.error('No active span in force_tracing')
        return
    span_to_trace.set_tag(opentracing.tags.SAMPLING_PRIORITY, 1)
    span_to_trace.set_baggage_item(SynapseBaggage.FORCE_TRACING, '1')

def is_context_forced_tracing(span_context: Optional['opentracing.SpanContext']) -> bool:
    if False:
        return 10
    'Check if sampling has been force for the given span context.'
    if span_context is None:
        return False
    return span_context.baggage.get(SynapseBaggage.FORCE_TRACING) is not None

@ensure_active_span('inject the span into a header dict')
def inject_header_dict(headers: Dict[bytes, List[bytes]], destination: Optional[str]=None, check_destination: bool=True) -> None:
    if False:
        print('Hello World!')
    "\n    Injects a span context into a dict of HTTP headers\n\n    Args:\n        headers: the dict to inject headers into\n        destination: address of entity receiving the span context. Must be given unless\n            check_destination is False. The context will only be injected if the\n            destination matches the opentracing whitelist\n        check_destination: If false, destination will be ignored and the context\n            will always be injected.\n\n    Note:\n        The headers set by the tracer are custom to the tracer implementation which\n        should be unique enough that they don't interfere with any headers set by\n        synapse or twisted. If we're still using jaeger these headers would be those\n        here:\n        https://github.com/jaegertracing/jaeger-client-python/blob/master/jaeger_client/constants.py\n    "
    if check_destination:
        if destination is None:
            raise ValueError('destination must be given unless check_destination is False')
        if not whitelisted_homeserver(destination):
            return
    span = opentracing.tracer.active_span
    carrier: Dict[str, str] = {}
    assert span is not None
    opentracing.tracer.inject(span.context, opentracing.Format.HTTP_HEADERS, carrier)
    for (key, value) in carrier.items():
        headers[key.encode()] = [value.encode()]

def inject_response_headers(response_headers: Headers) -> None:
    if False:
        print('Hello World!')
    'Inject the current trace id into the HTTP response headers'
    if not opentracing:
        return
    span = opentracing.tracer.active_span
    if not span:
        return
    trace_id = getattr(span, 'trace_id', None)
    if trace_id is not None:
        response_headers.addRawHeader('Synapse-Trace-Id', f'{trace_id:x}')

@ensure_active_span('get the active span context as a dict', ret=cast(Dict[str, str], {}))
def get_active_span_text_map(destination: Optional[str]=None) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    "\n    Gets a span context as a dict. This can be used instead of manually\n    injecting a span into an empty carrier.\n\n    Args:\n        destination: the name of the remote server.\n\n    Returns:\n        the active span's context if opentracing is enabled, otherwise empty.\n    "
    if destination and (not whitelisted_homeserver(destination)):
        return {}
    carrier: Dict[str, str] = {}
    assert opentracing.tracer.active_span is not None
    opentracing.tracer.inject(opentracing.tracer.active_span.context, opentracing.Format.TEXT_MAP, carrier)
    return carrier

@ensure_active_span('get the span context as a string.', ret={})
def active_span_context_as_string() -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n        The active span context encoded as a string.\n    '
    carrier: Dict[str, str] = {}
    if opentracing:
        assert opentracing.tracer.active_span is not None
        opentracing.tracer.inject(opentracing.tracer.active_span.context, opentracing.Format.TEXT_MAP, carrier)
    return json_encoder.encode(carrier)

def span_context_from_request(request: Request) -> 'Optional[opentracing.SpanContext]':
    if False:
        for i in range(10):
            print('nop')
    'Extract an opentracing context from the headers on an HTTP request\n\n    This is useful when we have received an HTTP request from another part of our\n    system, and want to link our spans to those of the remote system.\n    '
    if not opentracing:
        return None
    header_dict = {k.decode(): v[0].decode() for (k, v) in request.requestHeaders.getAllRawHeaders()}
    return opentracing.tracer.extract(opentracing.Format.HTTP_HEADERS, header_dict)

@only_if_tracing
def span_context_from_string(carrier: str) -> Optional['opentracing.SpanContext']:
    if False:
        while True:
            i = 10
    '\n    Returns:\n        The active span context decoded from a string.\n    '
    payload: Dict[str, str] = json_decoder.decode(carrier)
    return opentracing.tracer.extract(opentracing.Format.TEXT_MAP, payload)

@only_if_tracing
def extract_text_map(carrier: Dict[str, str]) -> Optional['opentracing.SpanContext']:
    if False:
        for i in range(10):
            print('nop')
    "\n    Wrapper method for opentracing's tracer.extract for TEXT_MAP.\n    Args:\n        carrier: a dict possibly containing a span context.\n\n    Returns:\n        The active span context extracted from carrier.\n    "
    return opentracing.tracer.extract(opentracing.Format.TEXT_MAP, carrier)

def _custom_sync_async_decorator(func: Callable[P, R], wrapping_logic: Callable[Concatenate[Callable[P, R], P], ContextManager[None]]) -> Callable[P, R]:
    if False:
        print('Hello World!')
    '\n    Decorates a function that is sync or async (coroutines), or that returns a Twisted\n    `Deferred`. The custom business logic of the decorator goes in `wrapping_logic`.\n\n    Example usage:\n    ```py\n    # Decorator to time the function and log it out\n    def duration(func: Callable[P, R]) -> Callable[P, R]:\n        @contextlib.contextmanager\n        def _wrapping_logic(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Generator[None, None, None]:\n            start_ts = time.time()\n            try:\n                yield\n            finally:\n                end_ts = time.time()\n                duration = end_ts - start_ts\n                logger.info("%s took %s seconds", func.__name__, duration)\n        return _custom_sync_async_decorator(func, _wrapping_logic)\n    ```\n\n    Args:\n        func: The function to be decorated\n        wrapping_logic: The business logic of your custom decorator.\n            This should be a ContextManager so you are able to run your logic\n            before/after the function as desired.\n    '
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            with wrapping_logic(func, *args, **kwargs):
                return await func(*args, **kwargs)
    else:

        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            if False:
                i = 10
                return i + 15
            scope = wrapping_logic(func, *args, **kwargs)
            scope.__enter__()
            try:
                result = func(*args, **kwargs)
                if isinstance(result, defer.Deferred):

                    def call_back(result: R) -> R:
                        if False:
                            i = 10
                            return i + 15
                        scope.__exit__(None, None, None)
                        return result

                    def err_back(result: R) -> R:
                        if False:
                            i = 10
                            return i + 15
                        scope.__exit__(None, None, None)
                        return result
                    result.addCallbacks(call_back, err_back)
                elif inspect.isawaitable(result):

                    async def wrap_awaitable() -> Any:
                        try:
                            assert isinstance(result, Awaitable)
                            awaited_result = await result
                            scope.__exit__(None, None, None)
                            return awaited_result
                        except Exception as e:
                            scope.__exit__(type(e), None, e.__traceback__)
                            raise
                    return wrap_awaitable()
                else:
                    scope.__exit__(None, None, None)
                return result
            except Exception as e:
                scope.__exit__(type(e), None, e.__traceback__)
                raise
    return _wrapper

def trace_with_opname(opname: str, *, tracer: Optional['opentracing.Tracer']=None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if False:
        return 10
    "\n    Decorator to trace a function with a custom opname.\n    See the module's doc string for usage examples.\n    "

    @contextlib.contextmanager
    def _wrapping_logic(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        with start_active_span(opname, tracer=tracer):
            yield

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        if False:
            print('Hello World!')
        if not opentracing:
            return func
        return _custom_sync_async_decorator(func, _wrapping_logic)
    return _decorator

def trace(func: Callable[P, R]) -> Callable[P, R]:
    if False:
        print('Hello World!')
    "\n    Decorator to trace a function.\n    Sets the operation name to that of the function's name.\n    See the module's doc string for usage examples.\n    "
    return trace_with_opname(func.__name__)(func)

def tag_args(func: Callable[P, R]) -> Callable[P, R]:
    if False:
        print('Hello World!')
    '\n    Decorator to tag all of the args to the active span.\n\n    Args:\n        func: `func` is assumed to be a method taking a `self` parameter, or a\n            `classmethod` taking a `cls` parameter. In either case, a tag is not\n            created for this parameter.\n    '
    if not opentracing:
        return func
    argspec = inspect.getfullargspec(func)

    @contextlib.contextmanager
    def _wrapping_logic(_func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        for (i, arg) in enumerate(args[1:], start=1):
            set_tag(SynapseTags.FUNC_ARG_PREFIX + argspec.args[i], str(arg))
        set_tag(SynapseTags.FUNC_ARGS, str(args[len(argspec.args):]))
        set_tag(SynapseTags.FUNC_KWARGS, str(kwargs))
        yield
    return _custom_sync_async_decorator(func, _wrapping_logic)

@contextlib.contextmanager
def trace_servlet(request: 'SynapseRequest', extract_context: bool=False) -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    'Returns a context manager which traces a request. It starts a span\n    with some servlet specific tags such as the request metrics name and\n    request information.\n\n    Args:\n        request\n        extract_context: Whether to attempt to extract the opentracing\n            context from the request the servlet is handling.\n    '
    if opentracing is None:
        yield
        return
    request_tags = {SynapseTags.REQUEST_ID: request.get_request_id(), tags.SPAN_KIND: tags.SPAN_KIND_RPC_SERVER, tags.HTTP_METHOD: request.get_method(), tags.HTTP_URL: request.get_redacted_uri(), tags.PEER_HOST_IPV6: request.get_client_ip_if_available()}
    request_name = request.request_metrics.name
    context = span_context_from_request(request) if extract_context else None
    scope = start_active_span(request_name, child_of=context, finish_on_close=False)
    request.set_opentracing_span(scope.span)
    with scope:
        inject_response_headers(request.responseHeaders)
        try:
            yield
        finally:
            scope.span.set_operation_name(request.request_metrics.name)
            request_tags[SynapseTags.REQUEST_TAG] = request.request_metrics.start_context.tag
            for (k, v) in request_tags.items():
                scope.span.set_tag(k, v)