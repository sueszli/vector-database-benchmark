import contextlib
import gc
import random
import sys
import warnings
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Hashable, Tuple
from weakref import WeakValueDictionary
import hypothesis.core
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.compat import GRAALPY, PYPY
if TYPE_CHECKING:
    from typing import Protocol

    class RandomLike(Protocol):
        seed: Callable[..., Any]
        getstate: Callable[[], Any]
        setstate: Callable[..., Any]
else:
    RandomLike = random.Random
_RKEY = count()
RANDOMS_TO_MANAGE: WeakValueDictionary = WeakValueDictionary({next(_RKEY): random})

class NumpyRandomWrapper:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        assert 'numpy' in sys.modules
        import numpy.random
        self.seed = numpy.random.seed
        self.getstate = numpy.random.get_state
        self.setstate = numpy.random.set_state
NP_RANDOM = None
if not (PYPY or GRAALPY):

    def _get_platform_base_refcount(r: Any) -> int:
        if False:
            print('Hello World!')
        return sys.getrefcount(r)
    _PLATFORM_REF_COUNT = _get_platform_base_refcount(object())
else:
    _PLATFORM_REF_COUNT = -1

def register_random(r: RandomLike) -> None:
    if False:
        while True:
            i = 10
    "Register (a weakref to) the given Random-like instance for management by\n    Hypothesis.\n\n    You can pass instances of structural subtypes of ``random.Random``\n    (i.e., objects with seed, getstate, and setstate methods) to\n    ``register_random(r)`` to have their states seeded and restored in the same\n    way as the global PRNGs from the ``random`` and ``numpy.random`` modules.\n\n    All global PRNGs, from e.g. simulation or scheduling frameworks, should\n    be registered to prevent flaky tests. Hypothesis will ensure that the\n    PRNG state is consistent for all test runs, always seeding them to zero and\n    restoring the previous state after the test, or, reproducibly varied if you\n    choose to use the :func:`~hypothesis.strategies.random_module` strategy.\n\n    ``register_random`` only makes `weakrefs\n    <https://docs.python.org/3/library/weakref.html#module-weakref>`_ to ``r``,\n    thus ``r`` will only be managed by Hypothesis as long as it has active\n    references elsewhere at runtime. The pattern ``register_random(MyRandom())``\n    will raise a ``ReferenceError`` to help protect users from this issue.\n    This check does not occur for the PyPy interpreter. See the following example for\n    an illustration of this issue\n\n    .. code-block:: python\n\n\n       def my_BROKEN_hook():\n           r = MyRandomLike()\n\n           # `r` will be garbage collected after the hook resolved\n           # and Hypothesis will 'forget' that it was registered\n           register_random(r)  # Hypothesis will emit a warning\n\n\n       rng = MyRandomLike()\n\n\n       def my_WORKING_hook():\n           register_random(rng)\n    "
    if not (hasattr(r, 'seed') and hasattr(r, 'getstate') and hasattr(r, 'setstate')):
        raise InvalidArgument(f'r={r!r} does not have all the required methods')
    if r in RANDOMS_TO_MANAGE.values():
        return
    if not (PYPY or GRAALPY):
        gc.collect()
        if not gc.get_referrers(r):
            if sys.getrefcount(r) <= _PLATFORM_REF_COUNT:
                raise ReferenceError(f'`register_random` was passed `r={r}` which will be garbage collected immediately after `register_random` creates a weakref to it. This will prevent Hypothesis from managing this PRNG. See the docs for `register_random` for more details.')
            else:
                warnings.warn('It looks like `register_random` was passed an object that could be garbage collected immediately after `register_random` creates a weakref to it. This will prevent Hypothesis from managing this PRNG. See the docs for `register_random` for more details.', HypothesisWarning, stacklevel=2)
    RANDOMS_TO_MANAGE[next(_RKEY)] = r

def get_seeder_and_restorer(seed: Hashable=0) -> Tuple[Callable[[], None], Callable[[], None]]:
    if False:
        for i in range(10):
            print('nop')
    'Return a pair of functions which respectively seed all and restore\n    the state of all registered PRNGs.\n\n    This is used by the core engine via `deterministic_PRNG`, and by users\n    via `register_random`.  We support registration of additional random.Random\n    instances (or other objects with seed, getstate, and setstate methods)\n    to force determinism on simulation or scheduling frameworks which avoid\n    using the global random state.  See e.g. #1709.\n    '
    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 32
    states: dict = {}
    if 'numpy' in sys.modules:
        global NP_RANDOM
        if NP_RANDOM is None:
            NP_RANDOM = RANDOMS_TO_MANAGE[next(_RKEY)] = NumpyRandomWrapper()

    def seed_all():
        if False:
            i = 10
            return i + 15
        assert not states
        for (k, r) in RANDOMS_TO_MANAGE.items():
            states[k] = r.getstate()
            r.seed(seed)

    def restore_all():
        if False:
            i = 10
            return i + 15
        for (k, state) in states.items():
            r = RANDOMS_TO_MANAGE.get(k)
            if r is not None:
                r.setstate(state)
        states.clear()
    return (seed_all, restore_all)

@contextlib.contextmanager
def deterministic_PRNG(seed=0):
    if False:
        print('Hello World!')
    'Context manager that handles random.seed without polluting global state.\n\n    See issue #1255 and PR #1295 for details and motivation - in short,\n    leaving the global pseudo-random number generator (PRNG) seeded is a very\n    bad idea in principle, and breaks all kinds of independence assumptions\n    in practice.\n    '
    if hypothesis.core._hypothesis_global_random is None:
        hypothesis.core._hypothesis_global_random = random.Random()
        register_random(hypothesis.core._hypothesis_global_random)
    (seed_all, restore_all) = get_seeder_and_restorer(seed)
    seed_all()
    try:
        yield
    finally:
        restore_all()