from typing import Optional, cast
from .errors import ActorNotFound
from .logging import get_logger
from .middleware import MiddlewareError, default_middleware
from .results import Results
global_broker: Optional['Broker'] = None

def get_broker() -> 'Broker':
    if False:
        i = 10
        return i + 15
    'Get the global broker instance.\n\n    If no global broker is set, a RabbitMQ broker will be returned.\n    If the RabbitMQ dependencies are not installed, a Redis broker\n    will be returned.\n\n    Returns:\n      Broker: The default Broker.\n    '
    global global_broker
    if global_broker is None:
        try:
            from .brokers.rabbitmq import RabbitmqBroker
            set_broker(RabbitmqBroker(host='127.0.0.1', port=5672, heartbeat=5, connection_attempts=5, blocked_connection_timeout=30))
        except ImportError:
            from .brokers.redis import RedisBroker
            set_broker(RedisBroker())
    global_broker = cast('Broker', global_broker)
    return global_broker

def set_broker(broker: 'Broker'):
    if False:
        for i in range(10):
            print('nop')
    'Configure the global broker instance.\n\n    Parameters:\n      broker(Broker): The broker instance to use by default.\n    '
    global global_broker
    global_broker = broker

class Broker:
    """Base class for broker implementations.

    Parameters:
      middleware(list[Middleware]): The set of middleware that apply
        to this broker.  If you supply this parameter, you are
        expected to declare *all* middleware.  Most of the time,
        you'll want to use :meth:`.add_middleware` instead.

    Attributes:
      actor_options(set[str]): The names of all the options actors may
        overwrite when they are declared.
    """

    def __init__(self, middleware=None):
        if False:
            for i in range(10):
                print('nop')
        self.logger = get_logger(__name__, type(self))
        self.actors = {}
        self.queues = {}
        self.delay_queues = set()
        self.actor_options = set()
        self.middleware = []
        if middleware is None:
            middleware = [m() for m in default_middleware]
        for m in middleware:
            self.add_middleware(m)

    def emit_before(self, signal, *args, **kwargs):
        if False:
            return 10
        signal = 'before_' + signal
        for middleware in self.middleware:
            try:
                getattr(middleware, signal)(self, *args, **kwargs)
            except MiddlewareError:
                raise
            except Exception:
                self.logger.critical('Unexpected failure in %s of %r.', signal, middleware, exc_info=True)

    def emit_after(self, signal, *args, **kwargs):
        if False:
            return 10
        signal = 'after_' + signal
        for middleware in reversed(self.middleware):
            try:
                getattr(middleware, signal)(self, *args, **kwargs)
            except Exception:
                self.logger.critical('Unexpected failure in %s of %r.', signal, middleware, exc_info=True)

    def add_middleware(self, middleware, *, before=None, after=None):
        if False:
            for i in range(10):
                print('nop')
        "Add a middleware object to this broker.  The middleware is\n        appended to the end of the middleware list by default.\n\n        You can specify another middleware (by class) as a reference\n        point for where the new middleware should be added.\n\n        Parameters:\n          middleware(Middleware): The middleware.\n          before(type): Add this middleware before a specific one.\n          after(type): Add this middleware after a specific one.\n\n        Raises:\n          ValueError: When either ``before`` or ``after`` refer to a\n            middleware that hasn't been registered yet.\n        "
        assert not (before and after), "provide either 'before' or 'after', but not both"
        if before or after:
            for (i, m) in enumerate(self.middleware):
                if isinstance(m, before or after):
                    break
            else:
                raise ValueError('Middleware %r not found' % (before or after))
            if before:
                self.middleware.insert(i, middleware)
            else:
                self.middleware.insert(i + 1, middleware)
        else:
            self.middleware.append(middleware)
        self.actor_options |= middleware.actor_options
        for actor_name in self.get_declared_actors():
            middleware.after_declare_actor(self, actor_name)
        for queue_name in self.get_declared_queues():
            middleware.after_declare_queue(self, queue_name)
        for queue_name in self.get_declared_delay_queues():
            middleware.after_declare_delay_queue(self, queue_name)

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close this broker and perform any necessary cleanup actions.\n        '

    def consume(self, queue_name, prefetch=1, timeout=30000):
        if False:
            return 10
        'Get an iterator that consumes messages off of the queue.\n\n        Raises:\n          QueueNotFound: If the given queue was never declared.\n\n        Parameters:\n          queue_name(str): The name of the queue to consume messages off of.\n          prefetch(int): The number of messages to prefetch per consumer.\n          timeout(int): The amount of time in milliseconds to idle for.\n\n        Returns:\n          Consumer: A message iterator.\n        '
        raise NotImplementedError

    def declare_actor(self, actor):
        if False:
            i = 10
            return i + 15
        'Declare a new actor on this broker.  Declaring an Actor\n        twice replaces the first actor with the second by name.\n\n        Parameters:\n          actor(Actor): The actor being declared.\n        '
        self.emit_before('declare_actor', actor)
        self.declare_queue(actor.queue_name)
        self.actors[actor.actor_name] = actor
        self.emit_after('declare_actor', actor)

    def declare_queue(self, queue_name):
        if False:
            for i in range(10):
                print('nop')
        'Declare a queue on this broker.  This method must be\n        idempotent.\n\n        Parameters:\n          queue_name(str): The name of the queue being declared.\n        '
        raise NotImplementedError

    def enqueue(self, message, *, delay=None):
        if False:
            print('Hello World!')
        'Enqueue a message on this broker.\n\n        Parameters:\n          message(Message): The message to enqueue.\n          delay(int): The number of milliseconds to delay the message for.\n\n        Returns:\n          Message: Either the original message or a copy of it.\n        '
        raise NotImplementedError

    def get_actor(self, actor_name):
        if False:
            print('Hello World!')
        'Look up an actor by its name.\n\n        Parameters:\n          actor_name(str): The name to look up.\n\n        Raises:\n          ActorNotFound: If the actor was never declared.\n\n        Returns:\n          Actor: The actor.\n        '
        try:
            return self.actors[actor_name]
        except KeyError:
            raise ActorNotFound(actor_name) from None

    def get_declared_actors(self):
        if False:
            print('Hello World!')
        'Get all declared actors.\n\n        Returns:\n          set[str]: The names of all the actors declared so far on\n          this Broker.\n        '
        return set(self.actors.keys())

    def get_declared_queues(self):
        if False:
            for i in range(10):
                print('nop')
        'Get all declared queues.\n\n        Returns:\n          set[str]: The names of all the queues declared so far on\n          this Broker.\n        '
        return set(self.queues.keys())

    def get_declared_delay_queues(self):
        if False:
            while True:
                i = 10
        'Get all declared delay queues.\n\n        Returns:\n          set[str]: The names of all the delay queues declared so far\n          on this Broker.\n        '
        return self.delay_queues.copy()

    def get_results_backend(self):
        if False:
            return 10
        "Get the backend of the Results middleware.\n\n        Raises:\n          RuntimeError: If the broker doesn't have a results backend.\n\n        Returns:\n          ResultBackend: The backend.\n        "
        for middleware in self.middleware:
            if isinstance(middleware, Results):
                return middleware.backend
        else:
            raise RuntimeError("The broker doesn't have a results backend.")

    def flush(self, queue_name):
        if False:
            for i in range(10):
                print('nop')
        'Drop all the messages from a queue.\n\n        Parameters:\n          queue_name(str): The name of the queue to flush.\n        '
        raise NotImplementedError()

    def flush_all(self):
        if False:
            i = 10
            return i + 15
        'Drop all messages from all declared queues.\n        '
        raise NotImplementedError()

    def join(self, queue_name, *, timeout=None):
        if False:
            return 10
        'Wait for all the messages on the given queue to be\n        processed.  This method is only meant to be used in tests to\n        wait for all the messages in a queue to be processed.\n\n        Subclasses that implement this function may add parameters.\n\n        Parameters:\n          queue_name(str): The queue to wait on.\n          timeout(Optional[int]): The max amount of time, in\n            milliseconds, to wait on this queue.\n        '
        raise NotImplementedError()

class Consumer:
    """Consumers iterate over messages on a queue.

    Consumers and their MessageProxies are *not* thread-safe.
    """

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Returns this instance as a Message iterator.\n        '
        return self

    def ack(self, message):
        if False:
            while True:
                i = 10
        'Acknowledge that a message has been processed, removing it\n        from the broker.\n\n        Parameters:\n          message(MessageProxy): The message to acknowledge.\n        '
        raise NotImplementedError

    def nack(self, message):
        if False:
            print('Hello World!')
        'Move a message to the dead-letter queue.\n\n        Parameters:\n          message(MessageProxy): The message to reject.\n        '
        raise NotImplementedError

    def requeue(self, messages):
        if False:
            while True:
                i = 10
        'Move unacked messages back to their queues.  This is called\n        by consumer threads when they fail or are shut down.  The\n        default implementation does nothing.\n\n        Parameters:\n          messages(list[MessageProxy]): The messages to requeue.\n        '

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        "Retrieve the next message off of the queue.  This method\n        blocks until a message becomes available.\n\n        Returns:\n          MessageProxy: A transparent proxy around a Message that can\n          be used to acknowledge or reject it once it's done being\n          processed.\n        "
        raise NotImplementedError

    def close(self):
        if False:
            print('Hello World!')
        'Close this consumer and perform any necessary cleanup actions.\n        '

class MessageProxy:
    """Base class for messages returned by :meth:`Broker.consume`.
    """

    def __init__(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.failed = False
        self._message = message
        self._exception = None

    def stuff_exception(self, exception):
        if False:
            print('Hello World!')
        'Stuff an exception into this message.\n        '
        self._exception = exception

    def clear_exception(self):
        if False:
            i = 10
            return i + 15
        'Remove the exception from this message.\n        '
        del self._exception

    def fail(self):
        if False:
            print('Hello World!')
        'Mark this message for rejection.\n        '
        self.failed = True

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self._message, name)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self._message)

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return True

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, MessageProxy):
            return self._message == other._message
        return self._message == other