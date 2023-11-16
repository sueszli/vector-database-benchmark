"""Task implementation: request context and the task base class."""
import sys
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu import serialization
from kombu.exceptions import OperationalError
from kombu.utils.uuid import uuid
from celery import current_app, states
from celery._state import _task_stack
from celery.canvas import _chain, group, signature
from celery.exceptions import Ignore, ImproperlyConfigured, MaxRetriesExceededError, Reject, Retry
from celery.local import class_property
from celery.result import EagerResult, denied_join_result
from celery.utils import abstract
from celery.utils.functional import mattrgetter, maybe_list
from celery.utils.imports import instantiate
from celery.utils.nodenames import gethostname
from celery.utils.serialization import raise_with_context
from .annotations import resolve_all as resolve_all_annotations
from .registry import _unpickle_task_v2
from .utils import appstr
__all__ = ('Context', 'Task')
extract_exec_options = mattrgetter('queue', 'routing_key', 'exchange', 'priority', 'expires', 'serializer', 'delivery_mode', 'compression', 'time_limit', 'soft_time_limit', 'immediate', 'mandatory')
R_BOUND_TASK = '<class {0.__name__} of {app}{flags}>'
R_UNBOUND_TASK = '<unbound {0.__name__}{flags}>'
R_INSTANCE = '<@task: {0.name} of {app}{flags}>'
TaskType = type

def _strflags(flags, default=''):
    if False:
        i = 10
        return i + 15
    if flags:
        return ' ({})'.format(', '.join(flags))
    return default

def _reprtask(task, fmt=None, flags=None):
    if False:
        for i in range(10):
            print('nop')
    flags = list(flags) if flags is not None else []
    flags.append('v2 compatible') if task.__v2_compat__ else None
    if not fmt:
        fmt = R_BOUND_TASK if task._app else R_UNBOUND_TASK
    return fmt.format(task, flags=_strflags(flags), app=appstr(task._app) if task._app else None)

class Context:
    """Task request variables (Task.request)."""
    _children = None
    _protected = 0
    args = None
    callbacks = None
    called_directly = True
    chain = None
    chord = None
    correlation_id = None
    delivery_info = None
    errbacks = None
    eta = None
    expires = None
    group = None
    group_index = None
    headers = None
    hostname = None
    id = None
    ignore_result = False
    is_eager = False
    kwargs = None
    logfile = None
    loglevel = None
    origin = None
    parent_id = None
    properties = None
    retries = 0
    reply_to = None
    replaced_task_nesting = 0
    root_id = None
    shadow = None
    taskset = None
    timelimit = None
    utc = None
    stamped_headers = None
    stamps = None

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.update(*args, **kwargs)
        if self.headers is None:
            self.headers = self._get_custom_headers(*args, **kwargs)

    def _get_custom_headers(self, *args, **kwargs):
        if False:
            return 10
        headers = {}
        headers.update(*args, **kwargs)
        celery_keys = {*Context.__dict__.keys(), 'lang', 'task', 'argsrepr', 'kwargsrepr'}
        for key in celery_keys:
            headers.pop(key, None)
        if not headers:
            return None
        return headers

    def update(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.__dict__.update(*args, **kwargs)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__.clear()

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, key, default)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<Context: {vars(self)!r}>'

    def as_execution_options(self):
        if False:
            i = 10
            return i + 15
        (limit_hard, limit_soft) = self.timelimit or (None, None)
        execution_options = {'task_id': self.id, 'root_id': self.root_id, 'parent_id': self.parent_id, 'group_id': self.group, 'group_index': self.group_index, 'shadow': self.shadow, 'chord': self.chord, 'chain': self.chain, 'link': self.callbacks, 'link_error': self.errbacks, 'expires': self.expires, 'soft_time_limit': limit_soft, 'time_limit': limit_hard, 'headers': self.headers, 'retries': self.retries, 'reply_to': self.reply_to, 'replaced_task_nesting': self.replaced_task_nesting, 'origin': self.origin}
        if hasattr(self, 'stamps') and hasattr(self, 'stamped_headers'):
            if self.stamps is not None and self.stamped_headers is not None:
                execution_options['stamped_headers'] = self.stamped_headers
                for (k, v) in self.stamps.items():
                    execution_options[k] = v
        return execution_options

    @property
    def children(self):
        if False:
            print('Hello World!')
        if self._children is None:
            self._children = []
        return self._children

@abstract.CallableTask.register
class Task:
    """Task base class.

    Note:
        When called tasks apply the :meth:`run` method.  This method must
        be defined by all tasks (that is unless the :meth:`__call__` method
        is overridden).
    """
    __trace__ = None
    __v2_compat__ = False
    MaxRetriesExceededError = MaxRetriesExceededError
    OperationalError = OperationalError
    Strategy = 'celery.worker.strategy:default'
    Request = 'celery.worker.request:Request'
    _app = None
    name = None
    typing = None
    max_retries = 3
    default_retry_delay = 3 * 60
    rate_limit = None
    ignore_result = None
    trail = True
    send_events = True
    store_errors_even_if_ignored = None
    serializer = None
    time_limit = None
    soft_time_limit = None
    backend = None
    track_started = None
    acks_late = None
    acks_on_failure_or_timeout = None
    reject_on_worker_lost = None
    throws = ()
    expires = None
    priority = None
    resultrepr_maxsize = 1024
    request_stack = None
    _default_request = None
    abstract = True
    _exec_options = None
    __bound__ = False
    from_config = (('serializer', 'task_serializer'), ('rate_limit', 'task_default_rate_limit'), ('priority', 'task_default_priority'), ('track_started', 'task_track_started'), ('acks_late', 'task_acks_late'), ('acks_on_failure_or_timeout', 'task_acks_on_failure_or_timeout'), ('reject_on_worker_lost', 'task_reject_on_worker_lost'), ('ignore_result', 'task_ignore_result'), ('store_eager_result', 'task_store_eager_result'), ('store_errors_even_if_ignored', 'task_store_errors_even_if_ignored'))
    _backend = None

    @classmethod
    def bind(cls, app):
        if False:
            print('Hello World!')
        (was_bound, cls.__bound__) = (cls.__bound__, True)
        cls._app = app
        conf = app.conf
        cls._exec_options = None
        if cls.typing is None:
            cls.typing = app.strict_typing
        for (attr_name, config_name) in cls.from_config:
            if getattr(cls, attr_name, None) is None:
                setattr(cls, attr_name, conf[config_name])
        if not was_bound:
            cls.annotate()
            from celery.utils.threads import LocalStack
            cls.request_stack = LocalStack()
        cls.on_bound(app)
        return app

    @classmethod
    def on_bound(cls, app):
        if False:
            while True:
                i = 10
        'Called when the task is bound to an app.\n\n        Note:\n            This class method can be defined to do additional actions when\n            the task class is bound to an app.\n        '

    @classmethod
    def _get_app(cls):
        if False:
            i = 10
            return i + 15
        if cls._app is None:
            cls._app = current_app
        if not cls.__bound__:
            cls.bind(cls._app)
        return cls._app
    app = class_property(_get_app, bind)

    @classmethod
    def annotate(cls):
        if False:
            i = 10
            return i + 15
        for d in resolve_all_annotations(cls.app.annotations, cls):
            for (key, value) in d.items():
                if key.startswith('@'):
                    cls.add_around(key[1:], value)
                else:
                    setattr(cls, key, value)

    @classmethod
    def add_around(cls, attr, around):
        if False:
            return 10
        orig = getattr(cls, attr)
        if getattr(orig, '__wrapped__', None):
            orig = orig.__wrapped__
        meth = around(orig)
        meth.__wrapped__ = orig
        setattr(cls, attr, meth)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        _task_stack.push(self)
        self.push_request(args=args, kwargs=kwargs)
        try:
            return self.run(*args, **kwargs)
        finally:
            self.pop_request()
            _task_stack.pop()

    def __reduce__(self):
        if False:
            while True:
                i = 10
        mod = type(self).__module__
        mod = mod if mod and mod in sys.modules else None
        return (_unpickle_task_v2, (self.name, mod), None)

    def run(self, *args, **kwargs):
        if False:
            return 10
        'The body of the task executed by workers.'
        raise NotImplementedError('Tasks must define the run method.')

    def start_strategy(self, app, consumer, **kwargs):
        if False:
            print('Hello World!')
        return instantiate(self.Strategy, self, app, consumer, **kwargs)

    def delay(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Star argument version of :meth:`apply_async`.\n\n        Does not support the extra options enabled by :meth:`apply_async`.\n\n        Arguments:\n            *args (Any): Positional arguments passed on to the task.\n            **kwargs (Any): Keyword arguments passed on to the task.\n        Returns:\n            celery.result.AsyncResult: Future promise.\n        '
        return self.apply_async(args, kwargs)

    def apply_async(self, args=None, kwargs=None, task_id=None, producer=None, link=None, link_error=None, shadow=None, **options):
        if False:
            for i in range(10):
                print('nop')
        "Apply tasks asynchronously by sending a message.\n\n        Arguments:\n            args (Tuple): The positional arguments to pass on to the task.\n\n            kwargs (Dict): The keyword arguments to pass on to the task.\n\n            countdown (float): Number of seconds into the future that the\n                task should execute.  Defaults to immediate execution.\n\n            eta (~datetime.datetime): Absolute time and date of when the task\n                should be executed.  May not be specified if `countdown`\n                is also supplied.\n\n            expires (float, ~datetime.datetime): Datetime or\n                seconds in the future for the task should expire.\n                The task won't be executed after the expiration time.\n\n            shadow (str): Override task name used in logs/monitoring.\n                Default is retrieved from :meth:`shadow_name`.\n\n            connection (kombu.Connection): Re-use existing broker connection\n                instead of acquiring one from the connection pool.\n\n            retry (bool): If enabled sending of the task message will be\n                retried in the event of connection loss or failure.\n                Default is taken from the :setting:`task_publish_retry`\n                setting.  Note that you need to handle the\n                producer/connection manually for this to work.\n\n            retry_policy (Mapping): Override the retry policy used.\n                See the :setting:`task_publish_retry_policy` setting.\n\n            time_limit (int): If set, overrides the default time limit.\n\n            soft_time_limit (int): If set, overrides the default soft\n                time limit.\n\n            queue (str, kombu.Queue): The queue to route the task to.\n                This must be a key present in :setting:`task_queues`, or\n                :setting:`task_create_missing_queues` must be\n                enabled.  See :ref:`guide-routing` for more\n                information.\n\n            exchange (str, kombu.Exchange): Named custom exchange to send the\n                task to.  Usually not used in combination with the ``queue``\n                argument.\n\n            routing_key (str): Custom routing key used to route the task to a\n                worker server.  If in combination with a ``queue`` argument\n                only used to specify custom routing keys to topic exchanges.\n\n            priority (int): The task priority, a number between 0 and 9.\n                Defaults to the :attr:`priority` attribute.\n\n            serializer (str): Serialization method to use.\n                Can be `pickle`, `json`, `yaml`, `msgpack` or any custom\n                serialization method that's been registered\n                with :mod:`kombu.serialization.registry`.\n                Defaults to the :attr:`serializer` attribute.\n\n            compression (str): Optional compression method\n                to use.  Can be one of ``zlib``, ``bzip2``,\n                or any custom compression methods registered with\n                :func:`kombu.compression.register`.\n                Defaults to the :setting:`task_compression` setting.\n\n            link (Signature): A single, or a list of tasks signatures\n                to apply if the task returns successfully.\n\n            link_error (Signature): A single, or a list of task signatures\n                to apply if an error occurs while executing the task.\n\n            producer (kombu.Producer): custom producer to use when publishing\n                the task.\n\n            add_to_parent (bool): If set to True (default) and the task\n                is applied while executing another task, then the result\n                will be appended to the parent tasks ``request.children``\n                attribute.  Trailing can also be disabled by default using the\n                :attr:`trail` attribute\n\n            ignore_result (bool): If set to `False` (default) the result\n                of a task will be stored in the backend. If set to `True`\n                the result will not be stored. This can also be set\n                using the :attr:`ignore_result` in the `app.task` decorator.\n\n            publisher (kombu.Producer): Deprecated alias to ``producer``.\n\n            headers (Dict): Message headers to be included in the message.\n\n        Returns:\n            celery.result.AsyncResult: Promise of future evaluation.\n\n        Raises:\n            TypeError: If not enough arguments are passed, or too many\n                arguments are passed.  Note that signature checks may\n                be disabled by specifying ``@task(typing=False)``.\n            kombu.exceptions.OperationalError: If a connection to the\n               transport cannot be made, or if the connection is lost.\n\n        Note:\n            Also supports all keyword arguments supported by\n            :meth:`kombu.Producer.publish`.\n        "
        if self.typing:
            try:
                check_arguments = self.__header__
            except AttributeError:
                pass
            else:
                check_arguments(*(args or ()), **kwargs or {})
        if self.__v2_compat__:
            shadow = shadow or self.shadow_name(self(), args, kwargs, options)
        else:
            shadow = shadow or self.shadow_name(args, kwargs, options)
        preopts = self._get_exec_options()
        options = dict(preopts, **options) if options else preopts
        options.setdefault('ignore_result', self.ignore_result)
        if self.priority:
            options.setdefault('priority', self.priority)
        app = self._get_app()
        if app.conf.task_always_eager:
            with app.producer_or_acquire(producer) as eager_producer:
                serializer = options.get('serializer')
                if serializer is None:
                    if eager_producer.serializer:
                        serializer = eager_producer.serializer
                    else:
                        serializer = app.conf.task_serializer
                body = (args, kwargs)
                (content_type, content_encoding, data) = serialization.dumps(body, serializer)
                (args, kwargs) = serialization.loads(data, content_type, content_encoding, accept=[content_type])
            with denied_join_result():
                return self.apply(args, kwargs, task_id=task_id or uuid(), link=link, link_error=link_error, **options)
        else:
            return app.send_task(self.name, args, kwargs, task_id=task_id, producer=producer, link=link, link_error=link_error, result_cls=self.AsyncResult, shadow=shadow, task_type=self, **options)

    def shadow_name(self, args, kwargs, options):
        if False:
            while True:
                i = 10
        "Override for custom task name in worker logs/monitoring.\n\n        Example:\n            .. code-block:: python\n\n                from celery.utils.imports import qualname\n\n                def shadow_name(task, args, kwargs, options):\n                    return qualname(args[0])\n\n                @app.task(shadow_name=shadow_name, serializer='pickle')\n                def apply_function_async(fun, *args, **kwargs):\n                    return fun(*args, **kwargs)\n\n        Arguments:\n            args (Tuple): Task positional arguments.\n            kwargs (Dict): Task keyword arguments.\n            options (Dict): Task execution options.\n        "

    def signature_from_request(self, request=None, args=None, kwargs=None, queue=None, **extra_options):
        if False:
            while True:
                i = 10
        request = self.request if request is None else request
        args = request.args if args is None else args
        kwargs = request.kwargs if kwargs is None else kwargs
        options = {**request.as_execution_options(), **extra_options}
        delivery_info = request.delivery_info or {}
        priority = delivery_info.get('priority')
        if priority is not None:
            options['priority'] = priority
        if queue:
            options['queue'] = queue
        else:
            exchange = delivery_info.get('exchange')
            routing_key = delivery_info.get('routing_key')
            if exchange == '' and routing_key:
                options['queue'] = routing_key
            else:
                options.update(delivery_info)
        return self.signature(args, kwargs, options, type=self, **extra_options)
    subtask_from_request = signature_from_request

    def retry(self, args=None, kwargs=None, exc=None, throw=True, eta=None, countdown=None, max_retries=None, **options):
        if False:
            print('Hello World!')
        'Retry the task, adding it to the back of the queue.\n\n        Example:\n            >>> from imaginary_twitter_lib import Twitter\n            >>> from proj.celery import app\n\n            >>> @app.task(bind=True)\n            ... def tweet(self, auth, message):\n            ...     twitter = Twitter(oauth=auth)\n            ...     try:\n            ...         twitter.post_status_update(message)\n            ...     except twitter.FailWhale as exc:\n            ...         # Retry in 5 minutes.\n            ...         raise self.retry(countdown=60 * 5, exc=exc)\n\n        Note:\n            Although the task will never return above as `retry` raises an\n            exception to notify the worker, we use `raise` in front of the\n            retry to convey that the rest of the block won\'t be executed.\n\n        Arguments:\n            args (Tuple): Positional arguments to retry with.\n            kwargs (Dict): Keyword arguments to retry with.\n            exc (Exception): Custom exception to report when the max retry\n                limit has been exceeded (default:\n                :exc:`~@MaxRetriesExceededError`).\n\n                If this argument is set and retry is called while\n                an exception was raised (``sys.exc_info()`` is set)\n                it will attempt to re-raise the current exception.\n\n                If no exception was raised it will raise the ``exc``\n                argument provided.\n            countdown (float): Time in seconds to delay the retry for.\n            eta (~datetime.datetime): Explicit time and date to run the\n                retry at.\n            max_retries (int): If set, overrides the default retry limit for\n                this execution.  Changes to this parameter don\'t propagate to\n                subsequent task retry attempts.  A value of :const:`None`,\n                means "use the default", so if you want infinite retries you\'d\n                have to set the :attr:`max_retries` attribute of the task to\n                :const:`None` first.\n            time_limit (int): If set, overrides the default time limit.\n            soft_time_limit (int): If set, overrides the default soft\n                time limit.\n            throw (bool): If this is :const:`False`, don\'t raise the\n                :exc:`~@Retry` exception, that tells the worker to mark\n                the task as being retried.  Note that this means the task\n                will be marked as failed if the task raises an exception,\n                or successful if it returns after the retry call.\n            **options (Any): Extra options to pass on to :meth:`apply_async`.\n\n        Raises:\n\n            celery.exceptions.Retry:\n                To tell the worker that the task has been re-sent for retry.\n                This always happens, unless the `throw` keyword argument\n                has been explicitly set to :const:`False`, and is considered\n                normal operation.\n        '
        request = self.request
        retries = request.retries + 1
        if max_retries is not None:
            self.override_max_retries = max_retries
        max_retries = self.max_retries if max_retries is None else max_retries
        if request.called_directly:
            raise_with_context(exc or Retry('Task can be retried', None))
        if not eta and countdown is None:
            countdown = self.default_retry_delay
        is_eager = request.is_eager
        S = self.signature_from_request(request, args, kwargs, countdown=countdown, eta=eta, retries=retries, **options)
        if max_retries is not None and retries > max_retries:
            if exc:
                raise_with_context(exc)
            raise self.MaxRetriesExceededError("Can't retry {}[{}] args:{} kwargs:{}".format(self.name, request.id, S.args, S.kwargs), task_args=S.args, task_kwargs=S.kwargs)
        ret = Retry(exc=exc, when=eta or countdown, is_eager=is_eager, sig=S)
        if is_eager:
            if throw:
                raise ret
            return ret
        try:
            S.apply_async()
        except Exception as exc:
            raise Reject(exc, requeue=False)
        if throw:
            raise ret
        return ret

    def apply(self, args=None, kwargs=None, link=None, link_error=None, task_id=None, retries=None, throw=None, logfile=None, loglevel=None, headers=None, **options):
        if False:
            return 10
        'Execute this task locally, by blocking until the task returns.\n\n        Arguments:\n            args (Tuple): positional arguments passed on to the task.\n            kwargs (Dict): keyword arguments passed on to the task.\n            throw (bool): Re-raise task exceptions.\n                Defaults to the :setting:`task_eager_propagates` setting.\n\n        Returns:\n            celery.result.EagerResult: pre-evaluated result.\n        '
        from celery.app.trace import build_tracer
        app = self._get_app()
        args = args or ()
        kwargs = kwargs or {}
        task_id = task_id or uuid()
        retries = retries or 0
        if throw is None:
            throw = app.conf.task_eager_propagates
        task = app._tasks[self.name]
        request = {'id': task_id, 'task': self.name, 'retries': retries, 'is_eager': True, 'logfile': logfile, 'loglevel': loglevel or 0, 'hostname': gethostname(), 'callbacks': maybe_list(link), 'errbacks': maybe_list(link_error), 'headers': headers, 'ignore_result': options.get('ignore_result', False), 'delivery_info': {'is_eager': True, 'exchange': options.get('exchange'), 'routing_key': options.get('routing_key'), 'priority': options.get('priority')}}
        if 'stamped_headers' in options:
            request['stamped_headers'] = maybe_list(options['stamped_headers'])
            request['stamps'] = {header: maybe_list(options.get(header, [])) for header in request['stamped_headers']}
        tb = None
        tracer = build_tracer(task.name, task, eager=True, propagate=throw, app=self._get_app())
        ret = tracer(task_id, args, kwargs, request)
        retval = ret.retval
        if isinstance(retval, ExceptionInfo):
            (retval, tb) = (retval.exception, retval.traceback)
            if isinstance(retval, ExceptionWithTraceback):
                retval = retval.exc
        if isinstance(retval, Retry) and retval.sig is not None:
            return retval.sig.apply(retries=retries + 1)
        state = states.SUCCESS if ret.info is None else ret.info.state
        return EagerResult(task_id, retval, state, traceback=tb, name=self.name)

    def AsyncResult(self, task_id, **kwargs):
        if False:
            print('Hello World!')
        'Get AsyncResult instance for the specified task.\n\n        Arguments:\n            task_id (str): Task id to get result for.\n        '
        return self._get_app().AsyncResult(task_id, backend=self.backend, task_name=self.name, **kwargs)

    def signature(self, args=None, *starargs, **starkwargs):
        if False:
            i = 10
            return i + 15
        'Create signature.\n\n        Returns:\n            :class:`~celery.signature`:  object for\n                this task, wrapping arguments and execution options\n                for a single task invocation.\n        '
        starkwargs.setdefault('app', self.app)
        return signature(self, args, *starargs, **starkwargs)
    subtask = signature

    def s(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create signature.\n\n        Shortcut for ``.s(*a, **k) -> .signature(a, k)``.\n        '
        return self.signature(args, kwargs)

    def si(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Create immutable signature.\n\n        Shortcut for ``.si(*a, **k) -> .signature(a, k, immutable=True)``.\n        '
        return self.signature(args, kwargs, immutable=True)

    def chunks(self, it, n):
        if False:
            for i in range(10):
                print('nop')
        'Create a :class:`~celery.canvas.chunks` task for this task.'
        from celery import chunks
        return chunks(self.s(), it, n, app=self.app)

    def map(self, it):
        if False:
            return 10
        'Create a :class:`~celery.canvas.xmap` task from ``it``.'
        from celery import xmap
        return xmap(self.s(), it, app=self.app)

    def starmap(self, it):
        if False:
            i = 10
            return i + 15
        'Create a :class:`~celery.canvas.xstarmap` task from ``it``.'
        from celery import xstarmap
        return xstarmap(self.s(), it, app=self.app)

    def send_event(self, type_, retry=True, retry_policy=None, **fields):
        if False:
            i = 10
            return i + 15
        'Send monitoring event message.\n\n        This can be used to add custom event types in :pypi:`Flower`\n        and other monitors.\n\n        Arguments:\n            type_ (str):  Type of event, e.g. ``"task-failed"``.\n\n        Keyword Arguments:\n            retry (bool):  Retry sending the message\n                if the connection is lost.  Default is taken from the\n                :setting:`task_publish_retry` setting.\n            retry_policy (Mapping): Retry settings.  Default is taken\n                from the :setting:`task_publish_retry_policy` setting.\n            **fields (Any): Map containing information about the event.\n                Must be JSON serializable.\n        '
        req = self.request
        if retry_policy is None:
            retry_policy = self.app.conf.task_publish_retry_policy
        with self.app.events.default_dispatcher(hostname=req.hostname) as d:
            return d.send(type_, uuid=req.id, retry=retry, retry_policy=retry_policy, **fields)

    def replace(self, sig):
        if False:
            i = 10
            return i + 15
        "Replace this task, with a new task inheriting the task id.\n\n        Execution of the host task ends immediately and no subsequent statements\n        will be run.\n\n        .. versionadded:: 4.0\n\n        Arguments:\n            sig (Signature): signature to replace with.\n            visitor (StampingVisitor): Visitor API object.\n\n        Raises:\n            ~@Ignore: This is always raised when called in asynchronous context.\n            It is best to always use ``return self.replace(...)`` to convey\n            to the reader that the task won't continue after being replaced.\n        "
        chord = self.request.chord
        if 'chord' in sig.options:
            raise ImproperlyConfigured('A signature replacing a task must not be part of a chord')
        if isinstance(sig, _chain) and (not getattr(sig, 'tasks', True)):
            raise ImproperlyConfigured('Cannot replace with an empty chain')
        if isinstance(sig, group):
            sig |= self.app.tasks['celery.accumulate'].s(index=0)
        for callback in maybe_list(self.request.callbacks) or []:
            sig.link(callback)
        for errback in maybe_list(self.request.errbacks) or []:
            sig.link_error(errback)
        if isinstance(sig, _chain) and 'link' in sig.options:
            final_task_links = sig.tasks[-1].options.setdefault('link', [])
            final_task_links.extend(maybe_list(sig.options['link']))
        sig.freeze(self.request.id)
        replaced_task_nesting = self.request.get('replaced_task_nesting', 0) + 1
        sig.set(chord=chord, group_id=self.request.group, group_index=self.request.group_index, root_id=self.request.root_id, replaced_task_nesting=replaced_task_nesting)
        if isinstance(sig, _chain):
            for chain_task in maybe_list(sig.tasks) or []:
                chain_task.set(replaced_task_nesting=replaced_task_nesting)
        for t in reversed(self.request.chain or []):
            chain_task = signature(t, app=self.app)
            chain_task.set(replaced_task_nesting=replaced_task_nesting)
            sig |= chain_task
        return self.on_replace(sig)

    def add_to_chord(self, sig, lazy=False):
        if False:
            print('Hello World!')
        "Add signature to the chord the current task is a member of.\n\n        .. versionadded:: 4.0\n\n        Currently only supported by the Redis result backend.\n\n        Arguments:\n            sig (Signature): Signature to extend chord with.\n            lazy (bool): If enabled the new task won't actually be called,\n                and ``sig.delay()`` must be called manually.\n        "
        if not self.request.chord:
            raise ValueError('Current task is not member of any chord')
        sig.set(group_id=self.request.group, group_index=self.request.group_index, chord=self.request.chord, root_id=self.request.root_id)
        result = sig.freeze()
        self.backend.add_to_chord(self.request.group, result)
        return sig.delay() if not lazy else sig

    def update_state(self, task_id=None, state=None, meta=None, **kwargs):
        if False:
            print('Hello World!')
        'Update task state.\n\n        Arguments:\n            task_id (str): Id of the task to update.\n                Defaults to the id of the current task.\n            state (str): New state.\n            meta (Dict): State meta-data.\n        '
        if task_id is None:
            task_id = self.request.id
        self.backend.store_result(task_id, meta, state, request=self.request, **kwargs)

    def before_start(self, task_id, args, kwargs):
        if False:
            i = 10
            return i + 15
        'Handler called before the task starts.\n\n        .. versionadded:: 5.2\n\n        Arguments:\n            task_id (str): Unique id of the task to execute.\n            args (Tuple): Original arguments for the task to execute.\n            kwargs (Dict): Original keyword arguments for the task to execute.\n\n        Returns:\n            None: The return value of this handler is ignored.\n        '

    def on_success(self, retval, task_id, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Success handler.\n\n        Run by the worker if the task executes successfully.\n\n        Arguments:\n            retval (Any): The return value of the task.\n            task_id (str): Unique id of the executed task.\n            args (Tuple): Original arguments for the executed task.\n            kwargs (Dict): Original keyword arguments for the executed task.\n\n        Returns:\n            None: The return value of this handler is ignored.\n        '

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        if False:
            while True:
                i = 10
        'Retry handler.\n\n        This is run by the worker when the task is to be retried.\n\n        Arguments:\n            exc (Exception): The exception sent to :meth:`retry`.\n            task_id (str): Unique id of the retried task.\n            args (Tuple): Original arguments for the retried task.\n            kwargs (Dict): Original keyword arguments for the retried task.\n            einfo (~billiard.einfo.ExceptionInfo): Exception information.\n\n        Returns:\n            None: The return value of this handler is ignored.\n        '

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if False:
            i = 10
            return i + 15
        'Error handler.\n\n        This is run by the worker when the task fails.\n\n        Arguments:\n            exc (Exception): The exception raised by the task.\n            task_id (str): Unique id of the failed task.\n            args (Tuple): Original arguments for the task that failed.\n            kwargs (Dict): Original keyword arguments for the task that failed.\n            einfo (~billiard.einfo.ExceptionInfo): Exception information.\n\n        Returns:\n            None: The return value of this handler is ignored.\n        '

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        if False:
            for i in range(10):
                print('nop')
        'Handler called after the task returns.\n\n        Arguments:\n            status (str): Current task state.\n            retval (Any): Task return value/exception.\n            task_id (str): Unique id of the task.\n            args (Tuple): Original arguments for the task.\n            kwargs (Dict): Original keyword arguments for the task.\n            einfo (~billiard.einfo.ExceptionInfo): Exception information.\n\n        Returns:\n            None: The return value of this handler is ignored.\n        '

    def on_replace(self, sig):
        if False:
            while True:
                i = 10
        'Handler called when the task is replaced.\n\n        Must return super().on_replace(sig) when overriding to ensure the task replacement\n        is properly handled.\n\n        .. versionadded:: 5.3\n\n        Arguments:\n            sig (Signature): signature to replace with.\n        '
        if self.request.is_eager:
            return sig.apply().get()
        else:
            sig.delay()
            raise Ignore('Replaced by new task')

    def add_trail(self, result):
        if False:
            return 10
        if self.trail:
            self.request.children.append(result)
        return result

    def push_request(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.request_stack.push(Context(*args, **kwargs))

    def pop_request(self):
        if False:
            while True:
                i = 10
        self.request_stack.pop()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '``repr(task)``.'
        return _reprtask(self, R_INSTANCE)

    def _get_request(self):
        if False:
            while True:
                i = 10
        'Get current request object.'
        req = self.request_stack.top
        if req is None:
            if self._default_request is None:
                self._default_request = Context()
            return self._default_request
        return req
    request = property(_get_request)

    def _get_exec_options(self):
        if False:
            print('Hello World!')
        if self._exec_options is None:
            self._exec_options = extract_exec_options(self)
        return self._exec_options

    @property
    def backend(self):
        if False:
            print('Hello World!')
        backend = self._backend
        if backend is None:
            return self.app.backend
        return backend

    @backend.setter
    def backend(self, value):
        if False:
            return 10
        self._backend = value

    @property
    def __name__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__
BaseTask = Task