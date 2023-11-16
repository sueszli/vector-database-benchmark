from collections import Counter
from contextlib import ExitStack
from .reentrant_messenger import ReentrantMessenger

class MarkovMessenger(ReentrantMessenger):
    """
    Markov dependency declaration.

    This is a statistical equivalent of a memory management arena.

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    :param int dim: An optional dimension to use for this independence index.
        Interface stub, behavior not yet implemented.
    :param str name: An optional unique name to help inference algorithms match
        :func:`pyro.markov` sites between models and guides.
        Interface stub, behavior not yet implemented.
    """

    def __init__(self, history=1, keep=False, dim=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        assert history >= 0
        self.history = history
        self.keep = keep
        self.dim = dim
        self.name = name
        if dim is not None:
            raise NotImplementedError('vectorized markov not yet implemented, try setting dim to None')
        if name is not None:
            raise NotImplementedError('vectorized markov not yet implemented, try setting name to None')
        self._iterable = None
        self._pos = -1
        self._stack = []
        super().__init__()

    def generator(self, iterable):
        if False:
            for i in range(10):
                print('nop')
        self._iterable = iterable
        return self

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._pos += 1
        if len(self._stack) <= self._pos:
            self._stack.append(set())
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self.keep:
            self._stack.pop()
        self._pos -= 1
        return super().__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if False:
            i = 10
            return i + 15
        if msg['done'] or type(msg['fn']).__name__ == '_Subsample':
            return
        infer = msg['infer']
        scope = infer.setdefault('_markov_scope', Counter())
        for pos in range(max(0, self._pos - self.history), self._pos + 1):
            scope.update(self._stack[pos])
        infer['_markov_depth'] = 1 + infer.get('_markov_depth', 0)
        self._stack[self._pos].add(msg['name'])