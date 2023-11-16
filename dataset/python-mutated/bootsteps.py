"""A directed acyclic graph of reusable components."""
from collections import deque
from threading import Event
from kombu.common import ignore_errors
from kombu.utils.encoding import bytes_to_str
from kombu.utils.imports import symbol_by_name
from .utils.graph import DependencyGraph, GraphFormatter
from .utils.imports import instantiate, qualname
from .utils.log import get_logger
try:
    from greenlet import GreenletExit
except ImportError:
    IGNORE_ERRORS = ()
else:
    IGNORE_ERRORS = (GreenletExit,)
__all__ = ('Blueprint', 'Step', 'StartStopStep', 'ConsumerStep')
RUN = 1
CLOSE = 2
TERMINATE = 3
logger = get_logger(__name__)

def _pre(ns, fmt):
    if False:
        while True:
            i = 10
    return f'| {ns.alias}: {fmt}'

def _label(s):
    if False:
        i = 10
        return i + 15
    return s.name.rsplit('.', 1)[-1]

class StepFormatter(GraphFormatter):
    """Graph formatter for :class:`Blueprint`."""
    blueprint_prefix = '⧉'
    conditional_prefix = '∘'
    blueprint_scheme = {'shape': 'parallelogram', 'color': 'slategray4', 'fillcolor': 'slategray3'}

    def label(self, step):
        if False:
            while True:
                i = 10
        return step and '{}{}'.format(self._get_prefix(step), bytes_to_str((step.label or _label(step)).encode('utf-8', 'ignore')))

    def _get_prefix(self, step):
        if False:
            for i in range(10):
                print('nop')
        if step.last:
            return self.blueprint_prefix
        if step.conditional:
            return self.conditional_prefix
        return ''

    def node(self, obj, **attrs):
        if False:
            while True:
                i = 10
        scheme = self.blueprint_scheme if obj.last else self.node_scheme
        return self.draw_node(obj, scheme, attrs)

    def edge(self, a, b, **attrs):
        if False:
            return 10
        if a.last:
            attrs.update(arrowhead='none', color='darkseagreen3')
        return self.draw_edge(a, b, self.edge_scheme, attrs)

class Blueprint:
    """Blueprint containing bootsteps that can be applied to objects.

    Arguments:
        steps Sequence[Union[str, Step]]: List of steps.
        name (str): Set explicit name for this blueprint.
        on_start (Callable): Optional callback applied after blueprint start.
        on_close (Callable): Optional callback applied before blueprint close.
        on_stopped (Callable): Optional callback applied after
            blueprint stopped.
    """
    GraphFormatter = StepFormatter
    name = None
    state = None
    started = 0
    default_steps = set()
    state_to_name = {0: 'initializing', RUN: 'running', CLOSE: 'closing', TERMINATE: 'terminating'}

    def __init__(self, steps=None, name=None, on_start=None, on_close=None, on_stopped=None):
        if False:
            return 10
        self.name = name or self.name or qualname(type(self))
        self.types = set(steps or []) | set(self.default_steps)
        self.on_start = on_start
        self.on_close = on_close
        self.on_stopped = on_stopped
        self.shutdown_complete = Event()
        self.steps = {}

    def start(self, parent):
        if False:
            for i in range(10):
                print('nop')
        self.state = RUN
        if self.on_start:
            self.on_start()
        for (i, step) in enumerate((s for s in parent.steps if s is not None)):
            self._debug('Starting %s', step.alias)
            self.started = i + 1
            step.start(parent)
            logger.debug('^-- substep ok')

    def human_state(self):
        if False:
            return 10
        return self.state_to_name[self.state or 0]

    def info(self, parent):
        if False:
            i = 10
            return i + 15
        info = {}
        for step in parent.steps:
            info.update(step.info(parent) or {})
        return info

    def close(self, parent):
        if False:
            i = 10
            return i + 15
        if self.on_close:
            self.on_close()
        self.send_all(parent, 'close', 'closing', reverse=False)

    def restart(self, parent, method='stop', description='restarting', propagate=False):
        if False:
            while True:
                i = 10
        self.send_all(parent, method, description, propagate=propagate)

    def send_all(self, parent, method, description=None, reverse=True, propagate=True, args=()):
        if False:
            while True:
                i = 10
        description = description or method.replace('_', ' ')
        steps = reversed(parent.steps) if reverse else parent.steps
        for step in steps:
            if step:
                fun = getattr(step, method, None)
                if fun is not None:
                    self._debug('%s %s...', description.capitalize(), step.alias)
                    try:
                        fun(parent, *args)
                    except Exception as exc:
                        if propagate:
                            raise
                        logger.exception('Error on %s %s: %r', description, step.alias, exc)

    def stop(self, parent, close=True, terminate=False):
        if False:
            print('Hello World!')
        what = 'terminating' if terminate else 'stopping'
        if self.state in (CLOSE, TERMINATE):
            return
        if self.state != RUN or self.started != len(parent.steps):
            self.state = TERMINATE
            self.shutdown_complete.set()
            return
        self.close(parent)
        self.state = CLOSE
        self.restart(parent, 'terminate' if terminate else 'stop', description=what, propagate=False)
        if self.on_stopped:
            self.on_stopped()
        self.state = TERMINATE
        self.shutdown_complete.set()

    def join(self, timeout=None):
        if False:
            print('Hello World!')
        try:
            self.shutdown_complete.wait(timeout=timeout)
        except IGNORE_ERRORS:
            pass

    def apply(self, parent, **kwargs):
        if False:
            while True:
                i = 10
        'Apply the steps in this blueprint to an object.\n\n        This will apply the ``__init__`` and ``include`` methods\n        of each step, with the object as argument::\n\n            step = Step(obj)\n            ...\n            step.include(obj)\n\n        For :class:`StartStopStep` the services created\n        will also be added to the objects ``steps`` attribute.\n        '
        self._debug('Preparing bootsteps.')
        order = self.order = []
        steps = self.steps = self.claim_steps()
        self._debug('Building graph...')
        for S in self._finalize_steps(steps):
            step = S(parent, **kwargs)
            steps[step.name] = step
            order.append(step)
        self._debug('New boot order: {%s}', ', '.join((s.alias for s in self.order)))
        for step in order:
            step.include(parent)
        return self

    def connect_with(self, other):
        if False:
            return 10
        self.graph.adjacent.update(other.graph.adjacent)
        self.graph.add_edge(type(other.order[0]), type(self.order[-1]))

    def __getitem__(self, name):
        if False:
            i = 10
            return i + 15
        return self.steps[name]

    def _find_last(self):
        if False:
            for i in range(10):
                print('nop')
        return next((C for C in self.steps.values() if C.last), None)

    def _firstpass(self, steps):
        if False:
            for i in range(10):
                print('nop')
        for step in steps.values():
            step.requires = [symbol_by_name(dep) for dep in step.requires]
        stream = deque((step.requires for step in steps.values()))
        while stream:
            for node in stream.popleft():
                node = symbol_by_name(node)
                if node.name not in self.steps:
                    steps[node.name] = node
                stream.append(node.requires)

    def _finalize_steps(self, steps):
        if False:
            i = 10
            return i + 15
        last = self._find_last()
        self._firstpass(steps)
        it = ((C, C.requires) for C in steps.values())
        G = self.graph = DependencyGraph(it, formatter=self.GraphFormatter(root=last))
        if last:
            for obj in G:
                if obj != last:
                    G.add_edge(last, obj)
        try:
            return G.topsort()
        except KeyError as exc:
            raise KeyError('unknown bootstep: %s' % exc)

    def claim_steps(self):
        if False:
            i = 10
            return i + 15
        return dict((self.load_step(step) for step in self.types))

    def load_step(self, step):
        if False:
            print('Hello World!')
        step = symbol_by_name(step)
        return (step.name, step)

    def _debug(self, msg, *args):
        if False:
            i = 10
            return i + 15
        return logger.debug(_pre(self, msg), *args)

    @property
    def alias(self):
        if False:
            print('Hello World!')
        return _label(self)

class StepType(type):
    """Meta-class for steps."""
    name = None
    requires = None

    def __new__(cls, name, bases, attrs):
        if False:
            i = 10
            return i + 15
        module = attrs.get('__module__')
        qname = f'{module}.{name}' if module else name
        attrs.update(__qualname__=qname, name=attrs.get('name') or qname)
        return super().__new__(cls, name, bases, attrs)

    def __str__(cls):
        if False:
            print('Hello World!')
        return cls.name

    def __repr__(cls):
        if False:
            return 10
        return 'step:{0.name}{{{0.requires!r}}}'.format(cls)

class Step(metaclass=StepType):
    """A Bootstep.

    The :meth:`__init__` method is called when the step
    is bound to a parent object, and can as such be used
    to initialize attributes in the parent object at
    parent instantiation-time.
    """
    name = None
    label = None
    conditional = False
    requires = ()
    last = False
    enabled = True

    def __init__(self, parent, **kwargs):
        if False:
            print('Hello World!')
        pass

    def include_if(self, parent):
        if False:
            while True:
                i = 10
        'Return true if bootstep should be included.\n\n        You can define this as an optional predicate that decides whether\n        this step should be created.\n        '
        return self.enabled

    def instantiate(self, name, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return instantiate(name, *args, **kwargs)

    def _should_include(self, parent):
        if False:
            i = 10
            return i + 15
        if self.include_if(parent):
            return (True, self.create(parent))
        return (False, None)

    def include(self, parent):
        if False:
            print('Hello World!')
        return self._should_include(parent)[0]

    def create(self, parent):
        if False:
            i = 10
            return i + 15
        'Create the step.'

    def __repr__(self):
        if False:
            return 10
        return f'<step: {self.alias}>'

    @property
    def alias(self):
        if False:
            while True:
                i = 10
        return self.label or _label(self)

    def info(self, obj):
        if False:
            for i in range(10):
                print('nop')
        pass

class StartStopStep(Step):
    """Bootstep that must be started and stopped in order."""
    obj = None

    def start(self, parent):
        if False:
            print('Hello World!')
        if self.obj:
            return self.obj.start()

    def stop(self, parent):
        if False:
            i = 10
            return i + 15
        if self.obj:
            return self.obj.stop()

    def close(self, parent):
        if False:
            i = 10
            return i + 15
        pass

    def terminate(self, parent):
        if False:
            print('Hello World!')
        if self.obj:
            return getattr(self.obj, 'terminate', self.obj.stop)()

    def include(self, parent):
        if False:
            for i in range(10):
                print('nop')
        (inc, ret) = self._should_include(parent)
        if inc:
            self.obj = ret
            parent.steps.append(self)
        return inc

class ConsumerStep(StartStopStep):
    """Bootstep that starts a message consumer."""
    requires = ('celery.worker.consumer:Connection',)
    consumers = None

    def get_consumers(self, channel):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('missing get_consumers')

    def start(self, c):
        if False:
            while True:
                i = 10
        channel = c.connection.channel()
        self.consumers = self.get_consumers(channel)
        for consumer in self.consumers or []:
            consumer.consume()

    def stop(self, c):
        if False:
            for i in range(10):
                print('nop')
        self._close(c, True)

    def shutdown(self, c):
        if False:
            i = 10
            return i + 15
        self._close(c, False)

    def _close(self, c, cancel_consumers=True):
        if False:
            print('Hello World!')
        channels = set()
        for consumer in self.consumers or []:
            if cancel_consumers:
                ignore_errors(c.connection, consumer.cancel)
            if consumer.channel:
                channels.add(consumer.channel)
        for channel in channels:
            ignore_errors(c.connection, channel.close)