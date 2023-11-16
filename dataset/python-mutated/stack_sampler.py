from __future__ import annotations
import threading
import timeit
import types
from contextvars import ContextVar
from typing import Any, Callable, List, NamedTuple, Optional
from pyinstrument.low_level.stat_profile import get_frame_info, setstatprofile
from pyinstrument.typing import LiteralStr
thread_locals = threading.local()
StackSamplerSubscriberTarget = Callable[[List[str], float, Optional['AsyncState']], None]

class StackSamplerSubscriber:

    def __init__(self, *, target: StackSamplerSubscriberTarget, desired_interval: float, bound_to_async_context: bool, async_state: AsyncState | None) -> None:
        if False:
            while True:
                i = 10
        self.target = target
        self.desired_interval = desired_interval
        self.bound_to_async_context = bound_to_async_context
        self.async_state = async_state
active_profiler_context_var: ContextVar[object | None] = ContextVar('active_profiler_context_var', default=None)

class StackSampler:
    """Manages setstatprofile for Profilers on a single thread"""
    subscribers: list[StackSamplerSubscriber]
    current_sampling_interval: float | None
    last_profile_time: float
    timer_func: Callable[[], float] | None

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.subscribers = []
        self.current_sampling_interval = None
        self.last_profile_time = 0.0
        self.timer_func = None

    def subscribe(self, target: StackSamplerSubscriberTarget, desired_interval: float, use_async_context: bool):
        if False:
            print('Hello World!')
        if use_async_context:
            if active_profiler_context_var.get() is not None:
                raise RuntimeError('There is already a profiler running. You cannot run multiple profilers in the same thread or async context, unless you disable async support.')
            active_profiler_context_var.set(target)
        self.subscribers.append(StackSamplerSubscriber(target=target, desired_interval=desired_interval, bound_to_async_context=use_async_context, async_state=AsyncState('in_context') if use_async_context else None))
        self._update()

    def unsubscribe(self, target: StackSamplerSubscriberTarget):
        if False:
            for i in range(10):
                print('nop')
        try:
            subscriber = next((s for s in self.subscribers if s.target == target))
        except StopIteration:
            raise StackSampler.SubscriberNotFound()
        if subscriber.bound_to_async_context:
            active_profiler_context_var.set(None)
        self.subscribers.remove(subscriber)
        self._update()

    def _update(self):
        if False:
            i = 10
            return i + 15
        if len(self.subscribers) == 0:
            self._stop_sampling()
            return
        min_subscribers_interval = min((s.desired_interval for s in self.subscribers))
        if self.current_sampling_interval != min_subscribers_interval:
            self._start_sampling(interval=min_subscribers_interval)

    def _start_sampling(self, interval: float):
        if False:
            i = 10
            return i + 15
        self.current_sampling_interval = interval
        if self.last_profile_time == 0.0:
            self.last_profile_time = self._timer()
        setstatprofile(self._sample, interval, active_profiler_context_var, self.timer_func)

    def _stop_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        setstatprofile(None)
        self.current_sampling_interval = None
        self.last_profile_time = 0.0

    def _sample(self, frame: types.FrameType, event: str, arg: Any):
        if False:
            while True:
                i = 10
        if event == 'context_changed':
            (new, old, coroutine_stack) = arg
            for subscriber in self.subscribers:
                if subscriber.target == old:
                    assert subscriber.bound_to_async_context
                    full_stack = build_call_stack(frame, event, arg)
                    if coroutine_stack:
                        full_stack.extend(reversed(coroutine_stack))
                        subscriber.async_state = AsyncState('out_of_context_awaited', info=full_stack)
                    else:
                        subscriber.async_state = AsyncState('out_of_context_unknown', info=full_stack)
                elif subscriber.target == new:
                    assert subscriber.bound_to_async_context
                    subscriber.async_state = AsyncState('in_context')
        else:
            now = self._timer()
            time_since_last_sample = now - self.last_profile_time
            call_stack = build_call_stack(frame, event, arg)
            for subscriber in self.subscribers:
                subscriber.target(call_stack, time_since_last_sample, subscriber.async_state)
            self.last_profile_time = now

    def _timer(self):
        if False:
            print('Hello World!')
        if self.timer_func:
            return self.timer_func()
        else:
            return timeit.default_timer()

    class SubscriberNotFound(Exception):
        pass

def get_stack_sampler() -> StackSampler:
    if False:
        while True:
            i = 10
    '\n    Gets the stack sampler for the current thread, creating it if necessary\n    '
    if not hasattr(thread_locals, 'stack_sampler'):
        thread_locals.stack_sampler = StackSampler()
    return thread_locals.stack_sampler

def build_call_stack(frame: types.FrameType | None, event: str, arg: Any) -> list[str]:
    if False:
        print('Hello World!')
    call_stack: list[str] = []
    if event == 'call':
        frame = frame.f_back if frame else None
    elif event == 'c_return' or event == 'c_exception':
        c_frame_identifier = '%s\x00%s\x00%i' % (getattr(arg, '__qualname__', arg.__name__), '<built-in>', 0)
        call_stack.append(c_frame_identifier)
    while frame is not None:
        call_stack.append(get_frame_info(frame))
        frame = frame.f_back
    thread = threading.current_thread()
    thread_identifier = '%s\x00%s\x00%i' % (thread.name, '<thread>', thread.ident)
    call_stack.append(thread_identifier)
    call_stack.reverse()
    return call_stack

class AsyncState(NamedTuple):
    state: LiteralStr['in_context', 'out_of_context_awaited', 'out_of_context_unknown']
    "\n    Definitions:\n      ``in_context``: indicates that the sample comes from the subscriber's\n      context.\n\n      ``out_of_context_awaited``: the sample comes from outside the\n      subscriber's context, but we tracked the await that happened before the\n      context exited. :attr:`info` contains the call stack of the await.\n\n      ``out_of_context_unknown``: the sample comes from outside the\n      subscriber's context, but the change of context didn't look like an\n      await. :attr:`info` contains the call stack when the context changed.\n    "
    info: Any = None