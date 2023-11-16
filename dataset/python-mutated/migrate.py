"""Message migration tools (Broker <-> Broker)."""
import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
__all__ = ('StopFiltering', 'State', 'republish', 'migrate_task', 'migrate_tasks', 'move', 'task_id_eq', 'task_id_in', 'start_filter', 'move_task_by_id', 'move_by_idmap', 'move_by_taskmap', 'move_direct', 'move_direct_by_id')
MOVING_PROGRESS_FMT = 'Moving task {state.filtered}/{state.strtotal}: {body[task]}[{body[id]}]'

class StopFiltering(Exception):
    """Semi-predicate used to signal filter stop."""

class State:
    """Migration progress state."""
    count = 0
    filtered = 0
    total_apx = 0

    @property
    def strtotal(self):
        if False:
            i = 10
            return i + 15
        if not self.total_apx:
            return '?'
        return str(self.total_apx)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.filtered:
            return f'^{self.filtered}'
        return f'{self.count}/{self.strtotal}'

def republish(producer, message, exchange=None, routing_key=None, remove_props=None):
    if False:
        while True:
            i = 10
    'Republish message.'
    if not remove_props:
        remove_props = ['application_headers', 'content_type', 'content_encoding', 'headers']
    body = ensure_bytes(message.body)
    (info, headers, props) = (message.delivery_info, message.headers, message.properties)
    exchange = info['exchange'] if exchange is None else exchange
    routing_key = info['routing_key'] if routing_key is None else routing_key
    (ctype, enc) = (message.content_type, message.content_encoding)
    compression = headers.pop('compression', None)
    expiration = props.pop('expiration', None)
    expiration = float(expiration) if expiration is not None else None
    for key in remove_props:
        props.pop(key, None)
    producer.publish(ensure_bytes(body), exchange=exchange, routing_key=routing_key, compression=compression, headers=headers, content_type=ctype, content_encoding=enc, expiration=expiration, **props)

def migrate_task(producer, body_, message, queues=None):
    if False:
        while True:
            i = 10
    'Migrate single task message.'
    info = message.delivery_info
    queues = {} if queues is None else queues
    republish(producer, message, exchange=queues.get(info['exchange']), routing_key=queues.get(info['routing_key']))

def filter_callback(callback, tasks):
    if False:
        while True:
            i = 10

    def filtered(body, message):
        if False:
            return 10
        if tasks and body['task'] not in tasks:
            return
        return callback(body, message)
    return filtered

def migrate_tasks(source, dest, migrate=migrate_task, app=None, queues=None, **kwargs):
    if False:
        return 10
    'Migrate tasks from one broker to another.'
    app = app_or_default(app)
    queues = prepare_queues(queues)
    producer = app.amqp.Producer(dest, auto_declare=False)
    migrate = partial(migrate, producer, queues=queues)

    def on_declare_queue(queue):
        if False:
            for i in range(10):
                print('nop')
        new_queue = queue(producer.channel)
        new_queue.name = queues.get(queue.name, queue.name)
        if new_queue.routing_key == queue.name:
            new_queue.routing_key = queues.get(queue.name, new_queue.routing_key)
        if new_queue.exchange.name == queue.name:
            new_queue.exchange.name = queues.get(queue.name, queue.name)
        new_queue.declare()
    return start_filter(app, source, migrate, queues=queues, on_declare_queue=on_declare_queue, **kwargs)

def _maybe_queue(app, q):
    if False:
        return 10
    if isinstance(q, str):
        return app.amqp.queues[q]
    return q

def move(predicate, connection=None, exchange=None, routing_key=None, source=None, app=None, callback=None, limit=None, transform=None, **kwargs):
    if False:
        print('Hello World!')
    "Find tasks by filtering them and move the tasks to a new queue.\n\n    Arguments:\n        predicate (Callable): Filter function used to decide the messages\n            to move.  Must accept the standard signature of ``(body, message)``\n            used by Kombu consumer callbacks.  If the predicate wants the\n            message to be moved it must return either:\n\n                1) a tuple of ``(exchange, routing_key)``, or\n\n                2) a :class:`~kombu.entity.Queue` instance, or\n\n                3) any other true value means the specified\n                    ``exchange`` and ``routing_key`` arguments will be used.\n        connection (kombu.Connection): Custom connection to use.\n        source: List[Union[str, kombu.Queue]]: Optional list of source\n            queues to use instead of the default (queues\n            in :setting:`task_queues`).  This list can also contain\n            :class:`~kombu.entity.Queue` instances.\n        exchange (str, kombu.Exchange): Default destination exchange.\n        routing_key (str): Default destination routing key.\n        limit (int): Limit number of messages to filter.\n        callback (Callable): Callback called after message moved,\n            with signature ``(state, body, message)``.\n        transform (Callable): Optional function to transform the return\n            value (destination) of the filter function.\n\n    Also supports the same keyword arguments as :func:`start_filter`.\n\n    To demonstrate, the :func:`move_task_by_id` operation can be implemented\n    like this:\n\n    .. code-block:: python\n\n        def is_wanted_task(body, message):\n            if body['id'] == wanted_id:\n                return Queue('foo', exchange=Exchange('foo'),\n                             routing_key='foo')\n\n        move(is_wanted_task)\n\n    or with a transform:\n\n    .. code-block:: python\n\n        def transform(value):\n            if isinstance(value, str):\n                return Queue(value, Exchange(value), value)\n            return value\n\n        move(is_wanted_task, transform=transform)\n\n    Note:\n        The predicate may also return a tuple of ``(exchange, routing_key)``\n        to specify the destination to where the task should be moved,\n        or a :class:`~kombu.entity.Queue` instance.\n        Any other true value means that the task will be moved to the\n        default exchange/routing_key.\n    "
    app = app_or_default(app)
    queues = [_maybe_queue(app, queue) for queue in source or []] or None
    with app.connection_or_acquire(connection, pool=False) as conn:
        producer = app.amqp.Producer(conn)
        state = State()

        def on_task(body, message):
            if False:
                while True:
                    i = 10
            ret = predicate(body, message)
            if ret:
                if transform:
                    ret = transform(ret)
                if isinstance(ret, Queue):
                    maybe_declare(ret, conn.default_channel)
                    (ex, rk) = (ret.exchange.name, ret.routing_key)
                else:
                    (ex, rk) = expand_dest(ret, exchange, routing_key)
                republish(producer, message, exchange=ex, routing_key=rk)
                message.ack()
                state.filtered += 1
                if callback:
                    callback(state, body, message)
                if limit and state.filtered >= limit:
                    raise StopFiltering()
        return start_filter(app, conn, on_task, consume_from=queues, **kwargs)

def expand_dest(ret, exchange, routing_key):
    if False:
        while True:
            i = 10
    try:
        (ex, rk) = ret
    except (TypeError, ValueError):
        (ex, rk) = (exchange, routing_key)
    return (ex, rk)

def task_id_eq(task_id, body, message):
    if False:
        i = 10
        return i + 15
    "Return true if task id equals task_id'."
    return body['id'] == task_id

def task_id_in(ids, body, message):
    if False:
        return 10
    "Return true if task id is member of set ids'."
    return body['id'] in ids

def prepare_queues(queues):
    if False:
        i = 10
        return i + 15
    if isinstance(queues, str):
        queues = queues.split(',')
    if isinstance(queues, list):
        queues = dict((tuple(islice(cycle(q.split(':')), None, 2)) for q in queues))
    if queues is None:
        queues = {}
    return queues

class Filterer:

    def __init__(self, app, conn, filter, limit=None, timeout=1.0, ack_messages=False, tasks=None, queues=None, callback=None, forever=False, on_declare_queue=None, consume_from=None, state=None, accept=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.app = app
        self.conn = conn
        self.filter = filter
        self.limit = limit
        self.timeout = timeout
        self.ack_messages = ack_messages
        self.tasks = set(str_to_list(tasks) or [])
        self.queues = prepare_queues(queues)
        self.callback = callback
        self.forever = forever
        self.on_declare_queue = on_declare_queue
        self.consume_from = [_maybe_queue(self.app, q) for q in consume_from or list(self.queues)]
        self.state = state or State()
        self.accept = accept

    def start(self):
        if False:
            while True:
                i = 10
        with self.prepare_consumer(self.create_consumer()):
            try:
                for _ in eventloop(self.conn, timeout=self.timeout, ignore_timeouts=self.forever):
                    pass
            except socket.timeout:
                pass
            except StopFiltering:
                pass
        return self.state

    def update_state(self, body, message):
        if False:
            print('Hello World!')
        self.state.count += 1
        if self.limit and self.state.count >= self.limit:
            raise StopFiltering()

    def ack_message(self, body, message):
        if False:
            i = 10
            return i + 15
        message.ack()

    def create_consumer(self):
        if False:
            for i in range(10):
                print('nop')
        return self.app.amqp.TaskConsumer(self.conn, queues=self.consume_from, accept=self.accept)

    def prepare_consumer(self, consumer):
        if False:
            i = 10
            return i + 15
        filter = self.filter
        update_state = self.update_state
        ack_message = self.ack_message
        if self.tasks:
            filter = filter_callback(filter, self.tasks)
            update_state = filter_callback(update_state, self.tasks)
            ack_message = filter_callback(ack_message, self.tasks)
        consumer.register_callback(filter)
        consumer.register_callback(update_state)
        if self.ack_messages:
            consumer.register_callback(self.ack_message)
        if self.callback is not None:
            callback = partial(self.callback, self.state)
            if self.tasks:
                callback = filter_callback(callback, self.tasks)
            consumer.register_callback(callback)
        self.declare_queues(consumer)
        return consumer

    def declare_queues(self, consumer):
        if False:
            for i in range(10):
                print('nop')
        for queue in consumer.queues:
            if self.queues and queue.name not in self.queues:
                continue
            if self.on_declare_queue is not None:
                self.on_declare_queue(queue)
            try:
                (_, mcount, _) = queue(consumer.channel).queue_declare(passive=True)
                if mcount:
                    self.state.total_apx += mcount
            except self.conn.channel_errors:
                pass

def start_filter(app, conn, filter, limit=None, timeout=1.0, ack_messages=False, tasks=None, queues=None, callback=None, forever=False, on_declare_queue=None, consume_from=None, state=None, accept=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Filter tasks.'
    return Filterer(app, conn, filter, limit=limit, timeout=timeout, ack_messages=ack_messages, tasks=tasks, queues=queues, callback=callback, forever=forever, on_declare_queue=on_declare_queue, consume_from=consume_from, state=state, accept=accept, **kwargs).start()

def move_task_by_id(task_id, dest, **kwargs):
    if False:
        while True:
            i = 10
    'Find a task by id and move it to another queue.\n\n    Arguments:\n        task_id (str): Id of task to find and move.\n        dest: (str, kombu.Queue): Destination queue.\n        transform (Callable): Optional function to transform the return\n            value (destination) of the filter function.\n        **kwargs (Any): Also supports the same keyword\n            arguments as :func:`move`.\n    '
    return move_by_idmap({task_id: dest}, **kwargs)

def move_by_idmap(map, **kwargs):
    if False:
        while True:
            i = 10
    "Move tasks by matching from a ``task_id: queue`` mapping.\n\n    Where ``queue`` is a queue to move the task to.\n\n    Example:\n        >>> move_by_idmap({\n        ...     '5bee6e82-f4ac-468e-bd3d-13e8600250bc': Queue('name'),\n        ...     'ada8652d-aef3-466b-abd2-becdaf1b82b3': Queue('name'),\n        ...     '3a2b140d-7db1-41ba-ac90-c36a0ef4ab1f': Queue('name')},\n        ...   queues=['hipri'])\n    "

    def task_id_in_map(body, message):
        if False:
            while True:
                i = 10
        return map.get(message.properties['correlation_id'])
    return move(task_id_in_map, limit=len(map), **kwargs)

def move_by_taskmap(map, **kwargs):
    if False:
        return 10
    "Move tasks by matching from a ``task_name: queue`` mapping.\n\n    ``queue`` is the queue to move the task to.\n\n    Example:\n        >>> move_by_taskmap({\n        ...     'tasks.add': Queue('name'),\n        ...     'tasks.mul': Queue('name'),\n        ... })\n    "

    def task_name_in_map(body, message):
        if False:
            while True:
                i = 10
        return map.get(body['task'])
    return move(task_name_in_map, **kwargs)

def filter_status(state, body, message, **kwargs):
    if False:
        print('Hello World!')
    print(MOVING_PROGRESS_FMT.format(state=state, body=body, **kwargs))
move_direct = partial(move, transform=worker_direct)
move_direct_by_id = partial(move_task_by_id, transform=worker_direct)
move_direct_by_idmap = partial(move_by_idmap, transform=worker_direct)
move_direct_by_taskmap = partial(move_by_taskmap, transform=worker_direct)