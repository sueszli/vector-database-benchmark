import inspect
import logging
from asyncio import AbstractEventLoop
from collections import defaultdict
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast
FuncT = TypeVar('FuncT', bound=Callable[..., Any])

class NotifierError(Exception):
    pass

class Notifier:
    """
    Allows communication between different Tribler modules and components.

    With Notifier, you can subscribe observer to a topic and receive notifications. The topic is a function,
    and the observer should be a callable with the same signature. Notifier is statically typed - if an observer
    has an incorrect signature or notification is called with wrong arguments, you should get a TypeError.
    PyCharm also should highlight incorrect observer registration and incorrect topic invocation.

    An example of usage:

    First, you need to create a Notifier instance. You can pass an event loop if the notifier should be able
    to process notifications asynchronously.

    >>> import asyncio
    >>> notifier = Notifier(loop=asyncio.get_event_loop())

    A topic is a function with an arbitrary signature. Usually, it has an empty body (a pass statement) but can include
    a debug code as well. It is called when notification is sent to observers.

    >>> def topic(foo: int, bar: str):
    ...     print("Notification is sent!")
    ...

    An observer should have the same signature as the topic (the return type is ignored for convenience). It may be a
    bound method of an object, in that case the `self` argument is also ignored.

    >>> def observer(foo: int, bar: str):
    ...     print("Observer called with", foo, bar)
    ...
    >>> def second_observer(foo: int, bar: str):
    ...     print("Second observer called with", foo, bar)
    ...

    To connect an observer to a specific notification, you can use the `add_observer` method. The method checks that
    the topic and the observer have the same signature.

    >>> notifier.add_observer(topic, observer)

    Observers can be registered as synchronous or asynchronous. Synchronous observers are called immediately,
    and asynchronous observers are called in the subsequent event loop iterations. By default, the observer
    is asynchronous if the notifier was initialized with an event loop. You can explicitly specify if the observer
    is synchronous or not:

    >>> notifier.add_observer(topic, second_observer, synchronous=True)

    To call observers for a specific topic in a type-safe manner, use square brackets syntax. If you are not aware
    what arguments should be used for specific topic, in IDE you can click on the topic function name and jump to the
    function signature.

    >>> notifier[topic](123, "abc")
    >>> notifier[topic](foo=123, bar="abc")

    When you invoke a notifier, all observers for the topic receive notification in the order as they were registered
    (synchronous observers first, then asynchronous).

    As an alternative, you can use the `notify` method, but without static type checks:

    >>> notifier.notify(topic, foo=123, bar="abc")

    The last way to invoke notification is by a topic function name. It can be useful when writing generic code.
    To be able to call the topic in this manner, it should have at least one observer:

    >>> notifier.notify_by_topic_name("topic", foo=123, bar="abc")

    You can also register a generic observer, receiving notifications for any topic. It will receive the topic
    as a first argument. When notification is called, generic observers are called before topic-specific observers
    in the same order as they were registered:

    >>> def generic_observer(topic, *args, **kwargs):
    ...     print("Generic observer called for", topic.__name__, "with", args, kwargs)
    ...
    >>> notifier.add_generic_observer(generic_observer)

    You can remove an observer or generic observer by calling the corresponding method:

    >>> notifier.remove_observer(observer)
    >>> notifier.remove_generic_observer(generic_observer)

    In Tribler, both Core and GUI have notifiers. Tribler uses generic observer to retranslate a subset of topics
    from Core to GUI. Core notifier is attached to the event loop and processes most topics asynchronously.
    GUI does not have an event loop, so GUI notifier processes retranslated topics synchronously. Basically, GUI
    notifier fires corresponding Qt signal for each topic.

    EventsEndpoint in Core and EventsRequestManager in GUI implement this logic of retranslation. EventsEndpoint adds
    a generic observer that listens to all topics, serializes a subset of notification calls to JSON, and sends them
    to GUI. EventRequestManager receives messages, deserializes arguments and calls `notifier.notify_by_topic_name`.
    """

    def __init__(self, loop: AbstractEventLoop=None):
        if False:
            for i in range(10):
                print('nop')
        self.lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.topics_by_name: Dict[str, Callable] = {}
        self.unknown_topic_names = set()
        self.topics: Dict[Callable, Dict[Callable, bool]] = defaultdict(dict)
        self.generic_observers: Dict[Callable, bool] = {}
        self.interceptors: Dict[Callable, bool] = {}
        self.loop = loop

    def add_observer(self, topic: FuncT, observer: FuncT, synchronous: Optional[bool]=None):
        if False:
            return 10
        ' Add the observer for the topic.\n        Each callback will be added no more than once. Callbacks are called in the same order as they were added.\n\n        topic:\n            A callable which represents a "topic" to subscribe\n\n        observer:\n            A callable which will be actually called when notification is sent to the topic\n\n        synchronous:\n            A strategy of how to call the observer. If True,\n\n\n        '
        synchronous = self._check_synchronous(synchronous)
        empty = inspect._empty
        topic_signature = inspect.signature(topic).replace(return_annotation=empty)
        callback_signature = inspect.signature(observer).replace(return_annotation=empty)
        if topic_signature != callback_signature:
            raise TypeError(f'Cannot add observer {observer!r} to topic "{topic.__name__}": the callback signature {callback_signature} does not match the topic signature {topic_signature}')
        if inspect.iscoroutinefunction(topic):
            raise TypeError(f'Topic cannot be a coroutine function. Got: {topic!r}')
        if inspect.iscoroutinefunction(observer):
            raise TypeError(f'Observer cannot be a coroutine function. Got: {observer!r}')
        if topic is observer:
            raise TypeError(f'Topic and observer cannot be the same function. Got: {topic!r}')
        self.logger.debug(f'Add observer topic {topic.__name__}')
        with self.lock:
            topic_name: str = topic.__name__
            prev_topic = self.topics_by_name.setdefault(topic_name, topic)
            if prev_topic is not topic:
                raise NotifierError(f'Cannot register topic {topic!r} because topic name {topic_name} is already taken by another topic {prev_topic!r}')
            prev_synchronous = self.topics[topic].setdefault(observer, synchronous)
            if prev_synchronous != synchronous:
                raise NotifierError('Cannot register the same observer with a different value of `synchronous` option')

    def _check_synchronous(self, synchronous: Optional[bool]) -> bool:
        if False:
            while True:
                i = 10
        if not any((synchronous is option for option in (True, False, None))):
            raise TypeError(f'`synchronous` option may be True, False or None. Got: {synchronous!r}')
        if synchronous is False and self.loop is None:
            raise TypeError('synchronous=False option cannot be specified for a notifier without an event loop')
        if synchronous is None:
            synchronous = not self.loop
        return synchronous

    def remove_observer(self, topic: FuncT, observer: FuncT):
        if False:
            while True:
                i = 10
        ' Remove the observer from the topic. In the case of a missed callback no error will be raised.\n        '
        with self.lock:
            observers = self.topics[topic]
            observers.pop(observer, None)
            comment = '' if not observers else f' (it still has {len(observers)} observers)'
        self.logger.debug(f'Remove observer {observer!r} from topic {topic.__name__}' + comment)

    def add_generic_observer(self, observer: Callable, synchronous: Optional[bool]=None):
        if False:
            while True:
                i = 10
        self.logger.debug(f'Add generic observer {observer!r}')
        with self.lock:
            self.generic_observers[observer] = self._check_synchronous(synchronous)

    def remove_generic_observer(self, observer: Callable):
        if False:
            while True:
                i = 10
        with self.lock:
            self.generic_observers.pop(observer, None)
        self.logger.debug(f'Remove generic observer {observer!r}')

    def __getitem__(self, topic: FuncT) -> FuncT:
        if False:
            i = 10
            return i + 15

        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            self.notify(topic, *args, **kwargs)
        return cast(FuncT, wrapper)

    def notify_by_topic_name(self, topic_name: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self.lock:
            topic = self.topics_by_name.get(topic_name)
        if topic is None:
            if topic_name not in self.unknown_topic_names:
                self.unknown_topic_names.add(topic_name)
                self.logger.warning(f'Topic with name `{topic_name}` not found')
        else:
            self.notify(topic, *args, **kwargs)

    def notify(self, topic: Callable, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Notify all observers about the topic.\n\n        Сan be called from any thread. Observers will be called from the reactor thread during the next iteration\n        of the event loop.  An exception when an observer is invoked will not affect other observers.\n        '
        self.logger.debug(f'Notification for topic {topic.__name__}')
        topic(*args, **kwargs)
        with self.lock:
            generic_observers: List[Tuple[Callable, bool]] = list(self.generic_observers.items())
            observers: List[Tuple[Callable, bool]] = list(self.topics[topic].items())
        generic_observer_args = (topic,) + args
        for (observer, synchronous) in generic_observers:
            if synchronous:
                self._notify(topic, observer, generic_observer_args, kwargs)
            else:
                self._notify_threadsafe(topic, observer, generic_observer_args, kwargs)
        for (observer, synchronous) in observers:
            if synchronous:
                self._notify(topic, observer, args, kwargs)
            else:
                self._notify_threadsafe(topic, observer, args, kwargs)

    def _notify_threadsafe(self, topic: Callable, observer: Callable, args: Tuple, kwargs: Dict[str, Any]):
        if False:
            print('Hello World!')
        try:
            self.loop.call_soon_threadsafe(self._notify, topic, observer, args, kwargs)
        except RuntimeError as e:
            self.logger.warning(e)

    def _notify(self, topic: Callable, observer: Callable, args: tuple, kwargs: dict):
        if False:
            print('Hello World!')
        self.logger.debug(f'Calling observer {observer!r} for topic {topic.__name__}')
        try:
            observer(*args, **kwargs)
        except Exception as e:
            self.logger.exception(e)