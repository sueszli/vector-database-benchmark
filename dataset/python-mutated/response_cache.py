import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Generic, Iterable, Optional, TypeVar
import attr
from twisted.internet import defer
from synapse.logging.context import make_deferred_yieldable, run_in_background
from synapse.logging.opentracing import active_span, start_active_span, start_active_span_follows_from
from synapse.util import Clock
from synapse.util.async_helpers import AbstractObservableDeferred, ObservableDeferred
from synapse.util.caches import EvictionReason, register_cache
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    import opentracing
KV = TypeVar('KV')
RV = TypeVar('RV')

@attr.s(auto_attribs=True)
class ResponseCacheContext(Generic[KV]):
    """Information about a missed ResponseCache hit

    This object can be passed into the callback for additional feedback
    """
    cache_key: KV
    'The cache key that caused the cache miss\n\n    This should be considered read-only.\n\n    TODO: in attrs 20.1, make it frozen with an on_setattr.\n    '
    should_cache: bool = True
    'Whether the result should be cached once the request completes.\n\n    This can be modified by the callback if it decides its result should not be cached.\n    '

@attr.s(auto_attribs=True)
class ResponseCacheEntry:
    result: AbstractObservableDeferred
    'The (possibly incomplete) result of the operation.\n\n    Note that we continue to store an ObservableDeferred even after the operation\n    completes (rather than switching to an immediate value), since that makes it\n    easier to cache Failure results.\n    '
    opentracing_span_context: 'Optional[opentracing.SpanContext]'
    'The opentracing span which generated/is generating the result'

class ResponseCache(Generic[KV]):
    """
    This caches a deferred response. Until the deferred completes it will be
    returned from the cache. This means that if the client retries the request
    while the response is still being computed, that original response will be
    used rather than trying to compute a new response.
    """

    def __init__(self, clock: Clock, name: str, timeout_ms: float=0):
        if False:
            print('Hello World!')
        self._result_cache: Dict[KV, ResponseCacheEntry] = {}
        self.clock = clock
        self.timeout_sec = timeout_ms / 1000.0
        self._name = name
        self._metrics = register_cache('response_cache', name, self, resizable=False)

    def size(self) -> int:
        if False:
            print('Hello World!')
        return len(self._result_cache)

    def __len__(self) -> int:
        if False:
            return 10
        return self.size()

    def keys(self) -> Iterable[KV]:
        if False:
            print('Hello World!')
        'Get the keys currently in the result cache\n\n        Returns both incomplete entries, and (if the timeout on this cache is non-zero),\n        complete entries which are still in the cache.\n\n        Note that the returned iterator is not safe in the face of concurrent execution:\n        behaviour is undefined if `wrap` is called during iteration.\n        '
        return self._result_cache.keys()

    def _get(self, key: KV) -> Optional[ResponseCacheEntry]:
        if False:
            for i in range(10):
                print('nop')
        'Look up the given key.\n\n        Args:\n            key: key to get in the cache\n\n        Returns:\n            The entry for this key, if any; else None.\n        '
        entry = self._result_cache.get(key)
        if entry is not None:
            self._metrics.inc_hits()
            return entry
        else:
            self._metrics.inc_misses()
            return None

    def _set(self, context: ResponseCacheContext[KV], deferred: 'defer.Deferred[RV]', opentracing_span_context: 'Optional[opentracing.SpanContext]') -> ResponseCacheEntry:
        if False:
            return 10
        'Set the entry for the given key to the given deferred.\n\n        *deferred* should run its callbacks in the sentinel logcontext (ie,\n        you should wrap normal synapse deferreds with\n        synapse.logging.context.run_in_background).\n\n        Args:\n            context: Information about the cache miss\n            deferred: The deferred which resolves to the result.\n            opentracing_span_context: An opentracing span wrapping the calculation\n\n        Returns:\n            The cache entry object.\n        '
        result = ObservableDeferred(deferred, consumeErrors=True)
        key = context.cache_key
        entry = ResponseCacheEntry(result, opentracing_span_context)
        self._result_cache[key] = entry

        def on_complete(r: RV) -> RV:
            if False:
                while True:
                    i = 10
            if self.timeout_sec and context.should_cache:
                self.clock.call_later(self.timeout_sec, self._entry_timeout, key)
            else:
                self.unset(key)
            return r
        result.addBoth(on_complete)
        return entry

    def unset(self, key: KV) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove the cached value for this key from the cache, if any.\n\n        Args:\n            key: key used to remove the cached value\n        '
        self._metrics.inc_evictions(EvictionReason.invalidation)
        self._result_cache.pop(key, None)

    def _entry_timeout(self, key: KV) -> None:
        if False:
            print('Hello World!')
        'For the call_later to remove from the cache'
        self._metrics.inc_evictions(EvictionReason.time)
        self._result_cache.pop(key, None)

    async def wrap(self, key: KV, callback: Callable[..., Awaitable[RV]], *args: Any, cache_context: bool=False, **kwargs: Any) -> RV:
        """Wrap together a *get* and *set* call, taking care of logcontexts

        First looks up the key in the cache, and if it is present makes it
        follow the synapse logcontext rules and returns it.

        Otherwise, makes a call to *callback(*args, **kwargs)*, which should
        follow the synapse logcontext rules, and adds the result to the cache.

        Example usage:

            async def handle_request(request):
                # etc
                return result

            result = await response_cache.wrap(
                key,
                handle_request,
                request,
            )

        Args:
            key: key to get/set in the cache

            callback: function to call if the key is not found in
                the cache

            *args: positional parameters to pass to the callback, if it is used

            cache_context: if set, the callback will be given a `cache_context` kw arg,
                which will be a ResponseCacheContext object.

            **kwargs: named parameters to pass to the callback, if it is used

        Returns:
            The result of the callback (from the cache, or otherwise)
        """
        entry = self._get(key)
        if not entry:
            logger.debug('[%s]: no cached result for [%s], calculating new one', self._name, key)
            context = ResponseCacheContext(cache_key=key)
            if cache_context:
                kwargs['cache_context'] = context
            span_context: Optional[opentracing.SpanContext] = None

            async def cb() -> RV:
                nonlocal span_context
                with start_active_span(f'ResponseCache[{self._name}].calculate'):
                    span = active_span()
                    if span:
                        span_context = span.context
                    return await callback(*args, **kwargs)
            d = run_in_background(cb)
            entry = self._set(context, d, span_context)
            return await make_deferred_yieldable(entry.result.observe())
        result = entry.result.observe()
        if result.called:
            logger.info('[%s]: using completed cached result for [%s]', self._name, key)
        else:
            logger.info('[%s]: using incomplete cached result for [%s]', self._name, key)
        span_context = entry.opentracing_span_context
        with start_active_span_follows_from(f'ResponseCache[{self._name}].wait', contexts=(span_context,) if span_context else ()):
            return await make_deferred_yieldable(result)