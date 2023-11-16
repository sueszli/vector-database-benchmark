"""

A `MessagePump` is a base class for any object which processes messages, which includes Widget, Screen, and App.

!!! tip

    Most of the method here are useful in general app development.

"""
from __future__ import annotations
import asyncio
import inspect
import threading
from asyncio import CancelledError, Queue, QueueEmpty, Task
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generator, Iterable, cast
from weakref import WeakSet
from . import Logger, events, log, messages
from ._asyncio import create_task
from ._callback import invoke
from ._context import NoActiveAppError, active_app, active_message_pump
from ._context import message_hook as message_hook_context_var
from ._context import prevent_message_types_stack
from ._on import OnNoWidget
from ._time import time
from .case import camel_to_snake
from .css.match import match
from .errors import DuplicateKeyHandlers
from .events import Event
from .message import Message
from .reactive import Reactive, TooManyComputesError
from .timer import Timer, TimerCallback
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from .app import App
    from .css.model import SelectorSet
Callback: TypeAlias = 'Callable[..., Any] | Callable[..., Awaitable[Any]]'

class CallbackError(Exception):
    pass

class MessagePumpClosed(Exception):
    pass

class _MessagePumpMeta(type):
    """Metaclass for message pump. This exists to populate a Message inner class of a Widget with the
    parent classes' name.
    """

    def __new__(cls, name: str, bases: tuple[type, ...], class_dict: dict[str, Any], **kwargs):
        if False:
            i = 10
            return i + 15
        namespace = camel_to_snake(name)
        isclass = inspect.isclass
        handlers: dict[type[Message], list[tuple[Callable, dict[str, tuple[SelectorSet, ...]]]]] = class_dict.get('_decorated_handlers', {})
        class_dict['_decorated_handlers'] = handlers
        for value in class_dict.values():
            if callable(value) and hasattr(value, '_textual_on'):
                textual_on: list[tuple[type[Message], dict[str, tuple[SelectorSet, ...]]]] = getattr(value, '_textual_on')
                for (message_type, selectors) in textual_on:
                    handlers.setdefault(message_type, []).append((value, selectors))
            if isclass(value) and issubclass(value, Message):
                if 'namespace' in value.__dict__:
                    value.handler_name = f"on_{value.__dict__['namespace']}_{camel_to_snake(value.__name__)}"
                else:
                    value.handler_name = f'on_{namespace}_{camel_to_snake(value.__name__)}'
        prefix = 'compute_'
        prefix_len = len(prefix)
        for (attr_name, value) in class_dict.items():
            if attr_name.startswith(prefix) and callable(value):
                reactive_name = attr_name[prefix_len:]
                if reactive_name in class_dict and isinstance(class_dict[reactive_name], Reactive) and (f'_{attr_name}' in class_dict):
                    raise TooManyComputesError(f"reactive {reactive_name!r} can't have two computes.")
        class_obj = super().__new__(cls, name, bases, class_dict, **kwargs)
        return class_obj

class MessagePump(metaclass=_MessagePumpMeta):
    """Base class which supplies a message pump."""

    def __init__(self, parent: MessagePump | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._message_queue: Queue[Message | None] = Queue()
        self._parent = parent
        self._running: bool = False
        self._closing: bool = False
        self._closed: bool = False
        self._disabled_messages: set[type[Message]] = set()
        self._pending_message: Message | None = None
        self._task: Task | None = None
        self._timers: WeakSet[Timer] = WeakSet()
        self._last_idle: float = time()
        self._max_idle: float | None = None
        self._mounted_event = asyncio.Event()
        self._next_callbacks: list[events.Callback] = []
        self._thread_id: int = threading.get_ident()

    @property
    def _prevent_message_types_stack(self) -> list[set[type[Message]]]:
        if False:
            i = 10
            return i + 15
        'The stack that manages prevented messages.'
        try:
            stack = prevent_message_types_stack.get()
        except LookupError:
            stack = [set()]
            prevent_message_types_stack.set(stack)
        return stack

    def _get_prevented_messages(self) -> set[type[Message]]:
        if False:
            i = 10
            return i + 15
        'A set of all the prevented message types.'
        return self._prevent_message_types_stack[-1]

    def _is_prevented(self, message_type: type[Message]) -> bool:
        if False:
            while True:
                i = 10
        'Check if a message type has been prevented via the\n        [prevent][textual.message_pump.MessagePump.prevent] context manager.\n\n        Args:\n            message_type: A message type.\n\n        Returns:\n            `True` if the message has been prevented from sending, or `False` if it will be sent as normal.\n        '
        return message_type in self._prevent_message_types_stack[-1]

    @contextmanager
    def prevent(self, *message_types: type[Message]) -> Generator[None, None, None]:
        if False:
            return 10
        'A context manager to *temporarily* prevent the given message types from being posted.\n\n        Example:\n            ```python\n            input = self.query_one(Input)\n            with self.prevent(Input.Changed):\n                input.value = "foo"\n            ```\n        '
        if message_types:
            prevent_stack = self._prevent_message_types_stack
            prevent_stack.append(prevent_stack[-1].union(message_types))
            try:
                yield
            finally:
                prevent_stack.pop()
        else:
            yield

    @property
    def task(self) -> Task:
        if False:
            while True:
                i = 10
        assert self._task is not None
        return self._task

    @property
    def has_parent(self) -> bool:
        if False:
            print('Hello World!')
        'Does this object have a parent?'
        return self._parent is not None

    @property
    def app(self) -> 'App[object]':
        if False:
            i = 10
            return i + 15
        '\n        Get the current app.\n\n        Returns:\n            The current app.\n\n        Raises:\n            NoActiveAppError: if no active app could be found for the current asyncio context\n        '
        try:
            return active_app.get()
        except LookupError:
            from .app import App
            node: MessagePump | None = self
            while not isinstance(node, App):
                if node is None:
                    raise NoActiveAppError()
                node = node._parent
            active_app.set(node)
            return node

    @property
    def is_parent_active(self) -> bool:
        if False:
            print('Hello World!')
        'Is the parent active?'
        return bool(self._parent and (not self._parent._closed) and (not self._parent._closing))

    @property
    def is_running(self) -> bool:
        if False:
            return 10
        'Is the message pump running (potentially processing messages)?'
        return self._running

    @property
    def log(self) -> Logger:
        if False:
            return 10
        'Get a logger for this object.\n\n        Returns:\n            A logger.\n        '
        return self.app._logger

    @property
    def is_attached(self) -> bool:
        if False:
            print('Hello World!')
        'Is the node attached to the app via the DOM?'
        from .app import App
        node = self
        while not isinstance(node, App):
            if node._parent is None:
                return False
            node = node._parent
        return True

    def _attach(self, parent: MessagePump) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the parent, and therefore attach this node to the tree.\n\n        Args:\n            parent: Parent node.\n        '
        self._parent = parent

    def _detach(self) -> None:
        if False:
            print('Hello World!')
        'Set the parent to None to remove the node from the tree.'
        self._parent = None

    def check_message_enabled(self, message: Message) -> bool:
        if False:
            while True:
                i = 10
        'Check if a given message is enabled (allowed to be sent).\n\n        Args:\n            message: A message object.\n\n        Returns:\n            `True` if the message will be sent, or `False` if it is disabled.\n        '
        return type(message) not in self._disabled_messages

    def disable_messages(self, *messages: type[Message]) -> None:
        if False:
            while True:
                i = 10
        'Disable message types from being processed.'
        self._disabled_messages.update(messages)

    def enable_messages(self, *messages: type[Message]) -> None:
        if False:
            return 10
        'Enable processing of messages types.'
        self._disabled_messages.difference_update(messages)

    async def _get_message(self) -> Message:
        """Get the next event on the queue, or None if queue is closed.

        Returns:
            Event object or None.
        """
        if self._closed:
            raise MessagePumpClosed('The message pump is closed')
        if self._pending_message is not None:
            try:
                return self._pending_message
            finally:
                self._pending_message = None
        message = await self._message_queue.get()
        if message is None:
            self._closed = True
            raise MessagePumpClosed('The message pump is now closed')
        return message

    def _peek_message(self) -> Message | None:
        if False:
            for i in range(10):
                print('nop')
        'Peek the message at the head of the queue (does not remove it from the queue),\n        or return None if the queue is empty.\n\n        Returns:\n            The message or None.\n        '
        if self._pending_message is None:
            try:
                message = self._message_queue.get_nowait()
            except QueueEmpty:
                pass
            else:
                if message is None:
                    self._closed = True
                    raise MessagePumpClosed('The message pump is now closed')
                self._pending_message = message
        if self._pending_message is not None:
            return self._pending_message
        return None

    def set_timer(self, delay: float, callback: TimerCallback | None=None, *, name: str | None=None, pause: bool=False) -> Timer:
        if False:
            i = 10
            return i + 15
        'Make a function call after a delay.\n\n        Args:\n            delay: Time to wait before invoking callback.\n            callback: Callback to call after time has expired.\n            name: Name of the timer (for debug).\n            pause: Start timer paused.\n\n        Returns:\n            A timer object.\n        '
        timer = Timer(self, delay, name=name or f'set_timer#{Timer._timer_count}', callback=callback, repeat=0, pause=pause)
        timer._start()
        self._timers.add(timer)
        return timer

    def set_interval(self, interval: float, callback: TimerCallback | None=None, *, name: str | None=None, repeat: int=0, pause: bool=False) -> Timer:
        if False:
            while True:
                i = 10
        'Call a function at periodic intervals.\n\n        Args:\n            interval: Time between calls.\n            callback: Function to call.\n            name: Name of the timer object.\n            repeat: Number of times to repeat the call or 0 for continuous.\n            pause: Start the timer paused.\n\n        Returns:\n            A timer object.\n        '
        timer = Timer(self, interval, name=name or f'set_interval#{Timer._timer_count}', callback=callback, repeat=repeat or None, pause=pause)
        timer._start()
        self._timers.add(timer)
        return timer

    def call_after_refresh(self, callback: Callback, *args: Any, **kwargs: Any) -> bool:
        if False:
            while True:
                i = 10
        'Schedule a callback to run after all messages are processed and the screen\n        has been refreshed. Positional and keyword arguments are passed to the callable.\n\n        Args:\n            callback: A callable.\n\n        Returns:\n            `True` if the callback was scheduled, or `False` if the callback could not be\n                scheduled (may occur if the message pump was closed or closing).\n\n        '
        message = messages.InvokeLater(partial(callback, *args, **kwargs))
        return self.post_message(message)

    def call_later(self, callback: Callback, *args: Any, **kwargs: Any) -> bool:
        if False:
            i = 10
            return i + 15
        'Schedule a callback to run after all messages are processed in this object.\n        Positional and keywords arguments are passed to the callable.\n\n        Args:\n            callback: Callable to call next.\n            *args: Positional arguments to pass to the callable.\n            **kwargs: Keyword arguments to pass to the callable.\n\n        Returns:\n            `True` if the callback was scheduled, or `False` if the callback could not be\n                scheduled (may occur if the message pump was closed or closing).\n\n        '
        message = events.Callback(callback=partial(callback, *args, **kwargs))
        return self.post_message(message)

    def call_next(self, callback: Callback, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Schedule a callback to run immediately after processing the current message.\n\n        Args:\n            callback: Callable to run after current event.\n            *args: Positional arguments to pass to the callable.\n            **kwargs: Keyword arguments to pass to the callable.\n        '
        callback_message = events.Callback(callback=partial(callback, *args, **kwargs))
        callback_message._prevent.update(self._get_prevented_messages())
        self._next_callbacks.append(callback_message)
        self.check_idle()

    def _on_invoke_later(self, message: messages.InvokeLater) -> None:
        if False:
            i = 10
            return i + 15
        self.app.screen._invoke_later(message.callback, message._sender or active_message_pump.get())

    def _close_messages_no_wait(self) -> None:
        if False:
            print('Hello World!')
        'Request the message queue to immediately exit.'
        self._message_queue.put_nowait(messages.CloseMessages())

    async def _on_close_messages(self, message: messages.CloseMessages) -> None:
        await self._close_messages()

    async def _close_messages(self, wait: bool=True) -> None:
        """Close message queue, and optionally wait for queue to finish processing."""
        if self._closed or self._closing:
            return
        self._closing = True
        stop_timers = list(self._timers)
        for timer in stop_timers:
            timer.stop()
        self._timers.clear()
        await self._message_queue.put(events.Unmount())
        Reactive._reset_object(self)
        await self._message_queue.put(None)
        if wait and self._task is not None and (asyncio.current_task() != self._task):
            try:
                running_widget = active_message_pump.get()
            except LookupError:
                running_widget = None
            if running_widget is None or running_widget is not self:
                await self._task

    def _start_messages(self) -> None:
        if False:
            print('Hello World!')
        'Start messages task.'
        if self.app._running:
            self._task = create_task(self._process_messages(), name=f'message pump {self}')
        else:
            self._closing = True
            self._closed = True

    async def _process_messages(self) -> None:
        self._running = True
        active_message_pump.set(self)
        if not await self._pre_process():
            self._running = False
            return
        try:
            await self._process_messages_loop()
        except CancelledError:
            pass
        finally:
            self._running = False
            for timer in list(self._timers):
                timer.stop()

    async def _pre_process(self) -> bool:
        """Procedure to run before processing messages.

        Returns:
            `True` if successful, or `False` if any exception occurred.

        """
        try:
            await self._dispatch_message(events.Compose())
            await self._dispatch_message(events.Mount())
            self.check_idle()
            self._post_mount()
        except Exception as error:
            self.app._handle_exception(error)
            return False
        finally:
            self._mounted_event.set()
        return True

    def _post_mount(self):
        if False:
            while True:
                i = 10
        'Called after the object has been mounted.'

    async def _process_messages_loop(self) -> None:
        """Process messages until the queue is closed."""
        _rich_traceback_guard = True
        self._thread_id = threading.get_ident()
        while not self._closed:
            try:
                message = await self._get_message()
            except MessagePumpClosed:
                break
            except CancelledError:
                raise
            except Exception as error:
                raise error from None
            while not (self._closed or self._closing):
                try:
                    pending = self._peek_message()
                except MessagePumpClosed:
                    break
                if pending is None or not message.can_replace(pending):
                    break
                try:
                    message = await self._get_message()
                except MessagePumpClosed:
                    break
            try:
                await self._dispatch_message(message)
            except CancelledError:
                raise
            except Exception as error:
                self._mounted_event.set()
                self.app._handle_exception(error)
                break
            finally:
                self._message_queue.task_done()
                current_time = time()
                if self._message_queue.empty() or (self._max_idle is not None and current_time - self._last_idle > self._max_idle):
                    self._last_idle = current_time
                    if not self._closed:
                        event = events.Idle()
                        for (_cls, method) in self._get_dispatch_methods('on_idle', event):
                            try:
                                await invoke(method, event)
                            except Exception as error:
                                self.app._handle_exception(error)
                                break
                    await self._flush_next_callbacks()

    async def _flush_next_callbacks(self) -> None:
        """Invoke pending callbacks in next callbacks queue."""
        callbacks = self._next_callbacks.copy()
        self._next_callbacks.clear()
        for callback in callbacks:
            try:
                await self._dispatch_message(callback)
            except Exception as error:
                self.app._handle_exception(error)
                break

    async def _dispatch_message(self, message: Message) -> None:
        """Dispatch a message received from the message queue.

        Args:
            message: A message object
        """
        _rich_traceback_guard = True
        if message.no_dispatch:
            return
        try:
            message_hook = message_hook_context_var.get()
        except LookupError:
            pass
        else:
            message_hook(message)
        with self.prevent(*message._prevent):
            if isinstance(message, Event):
                await self.on_event(message)
            else:
                await self._on_message(message)
            if self._next_callbacks:
                await self._flush_next_callbacks()

    def _get_dispatch_methods(self, method_name: str, message: Message) -> Iterable[tuple[type, Callable[[Message], Awaitable]]]:
        if False:
            for i in range(10):
                print('nop')
        'Gets handlers from the MRO\n\n        Args:\n            method_name: Handler method name.\n            message: Message object.\n        '
        from .widget import Widget
        methods_dispatched: set[Callable] = set()
        message_mro = [_type for _type in message.__class__.__mro__ if issubclass(_type, Message)]
        for cls in self.__class__.__mro__:
            if message._no_default_action:
                break
            decorated_handlers = cast('dict[type[Message], list[tuple[Callable, dict[str, tuple[SelectorSet, ...]]]]] | None', cls.__dict__.get('_decorated_handlers'))
            if decorated_handlers:
                for message_class in message_mro:
                    handlers = decorated_handlers.get(message_class, [])
                    for (method, selectors) in handlers:
                        if method in methods_dispatched:
                            continue
                        if not selectors:
                            yield (cls, method.__get__(self, cls))
                            methods_dispatched.add(method)
                        else:
                            if not message._sender:
                                continue
                            for (attribute, selector) in selectors.items():
                                node = getattr(message, attribute)
                                if not isinstance(node, Widget):
                                    raise OnNoWidget(f"on decorator can't match against {attribute!r} as it is not a widget.")
                                if not match(selector, node):
                                    break
                            else:
                                yield (cls, method.__get__(self, cls))
                                methods_dispatched.add(method)
            method = cls.__dict__.get(f'_{method_name}') or cls.__dict__.get(method_name)
            if method is not None and (not getattr(method, '_textual_on', None)):
                yield (cls, method.__get__(self, cls))

    async def on_event(self, event: events.Event) -> None:
        """Called to process an event.

        Args:
            event: An Event object.
        """
        await self._on_message(event)

    async def _on_message(self, message: Message) -> None:
        """Called to process a message.

        Args:
            message: A Message object.
        """
        _rich_traceback_guard = True
        handler_name = message.handler_name
        dispatched = False
        for (cls, method) in self._get_dispatch_methods(handler_name, message):
            log.event.verbosity(message.verbose)(message, '>>>', self, f'method=<{cls.__name__}.{handler_name}>')
            dispatched = True
            await invoke(method, message)
        if not dispatched:
            log.event.verbosity(message.verbose)(message, '>>>', self, 'method=None')
        if message.bubble and self._parent and (not message._stop_propagation):
            if message._sender is not None and message._sender == self._parent:
                message.stop()
            if self.is_parent_active and (not self._parent._closing):
                message._bubble_to(self._parent)

    def check_idle(self) -> None:
        if False:
            return 10
        'Prompt the message pump to call idle if the queue is empty.'
        if self._running and self._message_queue.empty():
            self.post_message(messages.Prompt())

    async def _post_message(self, message: Message) -> bool:
        """Post a message or an event to this message pump.

        This is an internal method for use where a coroutine is required.

        Args:
            message: A message object.

        Returns:
            True if the messages was posted successfully, False if the message was not posted
                (because the message pump was in the process of closing).
        """
        return self.post_message(message)

    def post_message(self, message: Message) -> bool:
        if False:
            print('Hello World!')
        "Posts a message on to this widget's queue.\n\n        Args:\n            message: A message (including Event).\n\n        Returns:\n            `True` if the messages was processed, `False` if it wasn't.\n        "
        _rich_traceback_omit = True
        if not hasattr(message, '_prevent'):
            raise RuntimeError('Message is missing attributes; did you forget to call super().__init__() ?')
        if self._closing or self._closed:
            return False
        if not self.check_message_enabled(message):
            return False
        message._prevent.update(self._get_prevented_messages())
        if self._thread_id != threading.get_ident() and self.app._loop is not None:
            loop = self.app._loop
            loop.call_soon_threadsafe(self._message_queue.put_nowait, message)
        else:
            self._message_queue.put_nowait(message)
        return True

    async def on_callback(self, event: events.Callback) -> None:
        await invoke(event.callback)

    async def dispatch_key(self, event: events.Key) -> bool:
        """Dispatch a key event to method.

        This method will call the method named 'key_<event.key>' if it exists.
        Some keys have aliases. The first alias found will be invoked if it exists.
        If multiple handlers exist that match the key, an exception is raised.

        Args:
            event: A key event.

        Returns:
            True if key was handled, otherwise False.

        Raises:
            DuplicateKeyHandlers: When there's more than 1 handler that could handle this key.
        """

        def get_key_handler(pump: MessagePump, key: str) -> Callable | None:
            if False:
                i = 10
                return i + 15
            'Look for the public and private handler methods by name on self.'
            public_handler_name = f'key_{key}'
            public_handler = getattr(pump, public_handler_name, None)
            private_handler_name = f'_key_{key}'
            private_handler = getattr(pump, private_handler_name, None)
            return public_handler or private_handler
        handled = False
        invoked_method = None
        key_name = event.name
        if not key_name:
            return False
        for key_method_name in event.name_aliases:
            key_method = get_key_handler(self, key_method_name)
            if key_method is not None:
                if invoked_method:
                    _raise_duplicate_key_handlers_error(key_name, invoked_method.__name__, key_method.__name__)
                handled = await invoke(key_method, event) is not False
                invoked_method = key_method
        return handled

    async def on_timer(self, event: events.Timer) -> None:
        event.prevent_default()
        event.stop()
        if event.callback is not None:
            try:
                await invoke(event.callback)
            except Exception as error:
                raise CallbackError(f'unable to run callback {event.callback!r}; {error}')

def _raise_duplicate_key_handlers_error(key_name: str, first_handler: str, second_handler: str) -> None:
    if False:
        print('Hello World!')
    'Raise exception for case where user presses a key and there are multiple candidate key handler methods for it.'
    raise DuplicateKeyHandlers(f"Multiple handlers for key press {key_name!r}.\nWe found both {first_handler!r} and {second_handler!r}, and didn't know which to call.\nConsider combining them into a single handler.")