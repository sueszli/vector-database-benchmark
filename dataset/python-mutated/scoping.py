"""
``pyro.contrib.autoname.scoping`` contains the implementation of
:func:`pyro.contrib.autoname.scope`, a tool for automatically appending
a semantically meaningful prefix to names of sample sites.
"""
import functools
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful

class NameCountMessenger(Messenger):
    """
    ``NameCountMessenger`` is the implementation of :func:`pyro.contrib.autoname.name_count`
    """

    def __enter__(self):
        if False:
            print('Hello World!')
        self._names = set()
        return super().__enter__()

    def _increment_name(self, name, label):
        if False:
            for i in range(10):
                print('nop')
        while (name, label) in self._names:
            split_name = name.split('__')
            if '__' in name and split_name[-1].isdigit():
                counter = int(split_name[-1]) + 1
                name = '__'.join(split_name[:-1] + [str(counter)])
            else:
                name = name + '__1'
        return name

    def _pyro_sample(self, msg):
        if False:
            print('Hello World!')
        msg['name'] = self._increment_name(msg['name'], 'sample')

    def _pyro_post_sample(self, msg):
        if False:
            while True:
                i = 10
        self._names.add((msg['name'], 'sample'))

    def _pyro_post_scope(self, msg):
        if False:
            print('Hello World!')
        self._names.add((msg['args'][0], 'scope'))

    def _pyro_scope(self, msg):
        if False:
            while True:
                i = 10
        msg['args'] = (self._increment_name(msg['args'][0], 'scope'),)

class ScopeMessenger(Messenger):
    """
    ``ScopeMessenger`` is the implementation of :func:`pyro.contrib.autoname.scope`
    """

    def __init__(self, prefix=None, inner=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.prefix = prefix
        self.inner = inner

    @staticmethod
    @effectful(type='scope')
    def _collect_scope(prefixed_scope):
        if False:
            while True:
                i = 10
        return prefixed_scope.split('/')[-1]

    def __enter__(self):
        if False:
            return 10
        if self.prefix is None:
            raise ValueError('no prefix was provided')
        if not self.inner:
            self.prefix = self._collect_scope(self.prefix)
        return super().__enter__()

    def __call__(self, fn):
        if False:
            print('Hello World!')
        if self.prefix is None:
            self.prefix = fn.__code__.co_name

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            with type(self)(prefix=self.prefix, inner=self.inner):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_scope(self, msg):
        if False:
            for i in range(10):
                print('nop')
        msg['args'] = ('{}/{}'.format(self.prefix, msg['args'][0]),)

    def _pyro_sample(self, msg):
        if False:
            for i in range(10):
                print('nop')
        msg['name'] = '{}/{}'.format(self.prefix, msg['name'])

def scope(fn=None, prefix=None, inner=None):
    if False:
        i = 10
        return i + 15
    '\n    :param fn: a stochastic function (callable containing Pyro primitive calls)\n    :param prefix: a string to prepend to sample names (optional if ``fn`` is provided)\n    :param inner: switch to determine where duplicate name counters appear\n    :returns: ``fn`` decorated with a :class:`~pyro.contrib.autoname.scoping.ScopeMessenger`\n\n    ``scope`` prepends a prefix followed by a ``/`` to the name at a Pyro sample site.\n    It works much like TensorFlow\'s ``name_scope`` and ``variable_scope``,\n    and can be used as a context manager, a decorator, or a higher-order function.\n\n    ``scope`` is very useful for aligning compositional models with guides or data.\n\n    Example::\n\n        >>> @scope(prefix="a")\n        ... def model():\n        ...     return pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "a/x" in poutine.trace(model).get_trace()\n\n\n    Example::\n\n        >>> def model():\n        ...     with scope(prefix="a"):\n        ...         return pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "a/x" in poutine.trace(model).get_trace()\n\n    Scopes compose as expected, with outer scopes appearing before inner scopes in names::\n\n        >>> @scope(prefix="b")\n        ... def model():\n        ...     with scope(prefix="a"):\n        ...         return pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "b/a/x" in poutine.trace(model).get_trace()\n\n    When used as a decorator or higher-order function,\n    ``scope`` will use the name of the input function as the prefix\n    if no user-specified prefix is provided.\n\n    Example::\n\n        >>> @scope\n        ... def model():\n        ...     return pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "model/x" in poutine.trace(model).get_trace()\n    '
    msngr = ScopeMessenger(prefix=prefix, inner=inner)
    return msngr(fn) if fn is not None else msngr

def name_count(fn=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    ``name_count`` is a very simple autonaming scheme that simply appends a suffix `"__"`\n    plus a counter to any name that appears multiple tims in an execution.\n    Only duplicate instances of a name get a suffix; the first instance is not modified.\n\n    Example::\n\n        >>> @name_count\n        ... def model():\n        ...     for i in range(3):\n        ...         pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "x" in poutine.trace(model).get_trace()\n        >>> assert "x__1" in poutine.trace(model).get_trace()\n        >>> assert "x__2" in poutine.trace(model).get_trace()\n\n    ``name_count`` also composes with :func:`~pyro.contrib.autoname.scope`\n    by adding a suffix to duplicate scope entrances:\n\n    Example::\n\n        >>> @name_count\n        ... def model():\n        ...     for i in range(3):\n        ...         with pyro.contrib.autoname.scope(prefix="a"):\n        ...             pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "a/x" in poutine.trace(model).get_trace()\n        >>> assert "a__1/x" in poutine.trace(model).get_trace()\n        >>> assert "a__2/x" in poutine.trace(model).get_trace()\n\n    Example::\n\n        >>> @name_count\n        ... def model():\n        ...     with pyro.contrib.autoname.scope(prefix="a"):\n        ...         for i in range(3):\n        ...             pyro.sample("x", dist.Bernoulli(0.5))\n        ...\n        >>> assert "a/x" in poutine.trace(model).get_trace()\n        >>> assert "a/x__1" in poutine.trace(model).get_trace()\n        >>> assert "a/x__2" in poutine.trace(model).get_trace()\n    '
    msngr = NameCountMessenger()
    return msngr(fn) if fn is not None else msngr