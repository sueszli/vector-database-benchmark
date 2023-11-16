from __future__ import annotations
import re
import time
from datetime import timedelta
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar, Union, overload
from .asyncio import async_to_sync
from .broker import Broker, get_broker
from .logging import get_logger
from .message import Message
if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    P = ParamSpec('P')
else:
    P = TypeVar('P')
_queue_name_re = re.compile('[a-zA-Z_][a-zA-Z0-9._-]*')
R = TypeVar('R')

class Actor(Generic[P, R]):
    """Thin wrapper around callables that stores metadata about how
    they should be executed asynchronously.  Actors are callable.

    Attributes:
      logger(Logger): The actor's logger.
      fn(callable): The underlying callable.
      broker(Broker): The broker this actor is bound to.
      actor_name(str): The actor's name.
      queue_name(str): The actor's queue.
      priority(int): The actor's priority.
      options(dict): Arbitrary options that are passed to the broker
        and middleware.
    """

    def __init__(self, fn: Callable[P, Union[R, Awaitable[R]]], *, broker: Broker, actor_name: str, queue_name: str, priority: int, options: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self.logger = get_logger(fn.__module__, actor_name)
        self.fn = async_to_sync(fn) if iscoroutinefunction(fn) else fn
        self.broker = broker
        self.actor_name = actor_name
        self.queue_name = queue_name
        self.priority = priority
        self.options = options
        self.broker.declare_actor(self)

    def message(self, *args: P.args, **kwargs: P.kwargs) -> Message[R]:
        if False:
            for i in range(10):
                print('nop')
        'Build a message.  This method is useful if you want to\n        compose actors.  See the actor composition documentation for\n        details.\n\n        Parameters:\n          *args(tuple): Positional arguments to send to the actor.\n          **kwargs: Keyword arguments to send to the actor.\n\n        Examples:\n          >>> (add.message(1, 2) | add.message(3))\n          pipeline([add(1, 2), add(3)])\n\n        Returns:\n          Message: A message that can be enqueued on a broker.\n        '
        return self.message_with_options(args=args, kwargs=kwargs)

    def message_with_options(self, *, args: tuple=(), kwargs: Optional[Dict[str, Any]]=None, **options) -> Message[R]:
        if False:
            i = 10
            return i + 15
        'Build a message with an arbitrary set of processing options.\n        This method is useful if you want to compose actors.  See the\n        actor composition documentation for details.\n\n        Parameters:\n          args(tuple): Positional arguments that are passed to the actor.\n          kwargs(dict): Keyword arguments that are passed to the actor.\n          **options: Arbitrary options that are passed to the\n            broker and any registered middleware.\n\n        Returns:\n          Message: A message that can be enqueued on a broker.\n        '
        for name in ['on_failure', 'on_success']:
            callback = options.get(name)
            if isinstance(callback, Actor):
                options[name] = callback.actor_name
            elif not isinstance(callback, (type(None), str)):
                raise TypeError(name + ' value must be an Actor')
        return Message(queue_name=self.queue_name, actor_name=self.actor_name, args=args, kwargs=kwargs or {}, options=options)

    def send(self, *args: P.args, **kwargs: P.kwargs) -> Message[R]:
        if False:
            print('Hello World!')
        'Asynchronously send a message to this actor.\n\n        Parameters:\n          *args: Positional arguments to send to the actor.\n          **kwargs: Keyword arguments to send to the actor.\n\n        Returns:\n          Message: The enqueued message.\n        '
        return self.send_with_options(args=args, kwargs=kwargs)

    def send_with_options(self, *, args: tuple=(), kwargs: Optional[Dict[str, Any]]=None, delay: Optional[timedelta | int]=None, **options) -> Message[R]:
        if False:
            while True:
                i = 10
        'Asynchronously send a message to this actor, along with an\n        arbitrary set of processing options for the broker and\n        middleware.\n\n        Parameters:\n          args(tuple): Positional arguments that are passed to the actor.\n          kwargs(dict): Keyword arguments that are passed to the actor.\n          delay(int): The minimum amount of time, in milliseconds, the\n            message should be delayed by. Also accepts a timedelta.\n          **options: Arbitrary options that are passed to the\n            broker and any registered middleware.\n\n        Returns:\n          Message: The enqueued message.\n        '
        if isinstance(delay, timedelta):
            delay = delay.total_seconds() * 1000
        message = self.message_with_options(args=args, kwargs=kwargs, **options)
        return self.broker.enqueue(message, delay=delay)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if False:
            return 10
        'Synchronously call this actor.\n\n        Parameters:\n          *args: Positional arguments to send to the actor.\n          **kwargs: Keyword arguments to send to the actor.\n\n        Returns:\n          Whatever the underlying function backing this actor returns.\n        '
        try:
            self.logger.debug('Received args=%r kwargs=%r.', args, kwargs)
            start = time.perf_counter()
            return self.fn(*args, **kwargs)
        finally:
            delta = time.perf_counter() - start
            self.logger.debug('Completed after %.02fms.', delta * 1000)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Actor(%(fn)r, queue_name=%(queue_name)r, actor_name=%(actor_name)r)' % vars(self)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Actor(%(actor_name)s)' % vars(self)

@overload
def actor(fn: Callable[P, R], **kwargs) -> Actor[P, R]:
    if False:
        return 10
    pass

@overload
def actor(fn: None=None, **kwargs) -> Callable[[Callable[P, R]], Actor[P, R]]:
    if False:
        return 10
    pass

def actor(fn: Optional[Callable[P, R]]=None, *, actor_class: Callable[..., Actor[P, R]]=Actor, actor_name: Optional[str]=None, queue_name: str='default', priority: int=0, broker: Optional[Broker]=None, **options) -> Union[Actor[P, R], Callable]:
    if False:
        while True:
            i = 10
    "Declare an actor.\n\n    Examples:\n\n      >>> import dramatiq\n\n      >>> @dramatiq.actor\n      ... def add(x, y):\n      ...     print(x + y)\n      ...\n      >>> add\n      Actor(<function add at 0x106c6d488>, queue_name='default', actor_name='add')\n\n      >>> add(1, 2)\n      3\n\n      >>> add.send(1, 2)\n      Message(\n        queue_name='default',\n        actor_name='add',\n        args=(1, 2), kwargs={}, options={},\n        message_id='e0d27b45-7900-41da-bb97-553b8a081206',\n        message_timestamp=1497862448685)\n\n    Parameters:\n      fn(callable): The function to wrap.\n      actor_class(type): Type created by the decorator.  Defaults to\n        :class:`Actor` but can be any callable as long as it returns an\n        actor and takes the same arguments as the :class:`Actor` class.\n      actor_name(str): The name of the actor.\n      queue_name(str): The name of the queue to use.\n      priority(int): The actor's global priority.  If two tasks have\n        been pulled on a worker concurrently and one has a higher\n        priority than the other then it will be processed first.\n        Lower numbers represent higher priorities.\n      broker(Broker): The broker to use with this actor.\n      **options: Arbitrary options that vary with the set of\n        middleware that you use.  See ``get_broker().actor_options``.\n\n    Returns:\n      Actor: The decorated function.\n    "

    def decorator(fn: Callable[..., R]) -> Actor[P, R]:
        if False:
            return 10
        nonlocal actor_name, broker
        actor_name = actor_name or fn.__name__
        if not _queue_name_re.fullmatch(queue_name):
            raise ValueError('Queue names must start with a letter or an underscore followed by any number of letters, digits, dashes or underscores.')
        broker = broker or get_broker()
        invalid_options = set(options) - broker.actor_options
        if invalid_options:
            invalid_options_list = ', '.join(invalid_options)
            raise ValueError('The following actor options are undefined: %s. Did you forget to add a middleware to your Broker?' % invalid_options_list)
        return actor_class(fn, actor_name=actor_name, queue_name=queue_name, priority=priority, broker=broker, options=options)
    if fn is None:
        return decorator
    return decorator(fn)