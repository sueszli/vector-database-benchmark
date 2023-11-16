""" Provide a set of decorators useful for repeatedly updating a
a function parameter in a specified way each time the function is
called.

These decorators can be especially useful in conjunction with periodic
callbacks in a Bokeh server application.

Example:

    As an example, consider the ``bounce`` forcing function, which
    advances a sequence forwards and backwards:

    .. code-block:: python

        from bokeh.driving import bounce

        @bounce([0, 1, 2])
        def update(i):
            print(i)

    If this function is repeatedly called, it will print the following
    sequence on standard out:

    .. code-block:: none

        0 1 2 2 1 0 0 1 2 2 1 ...

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar
__all__ = ('bounce', 'cosine', 'count', 'force', 'linear', 'repeat', 'sine')

def bounce(sequence: Sequence[int]) -> partial[Callable[[], None]]:
    if False:
        return 10
    ' Return a driver function that can advance a "bounced" sequence\n    of values.\n\n    .. code-block:: none\n\n        seq = [0, 1, 2, 3]\n\n        # bounce(seq) => [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, ...]\n\n    Args:\n        sequence (seq) : a sequence of values for the driver to bounce\n\n    '
    N = len(sequence)

    def f(i: int) -> int:
        if False:
            return 10
        (div, mod) = divmod(i, N)
        if div % 2 == 0:
            return sequence[mod]
        else:
            return sequence[N - mod - 1]
    return partial(force, sequence=_advance(f))

def cosine(w: float, A: float=1, phi: float=0, offset: float=0) -> partial[Callable[[], None]]:
    if False:
        i = 10
        return i + 15
    ' Return a driver function that can advance a sequence of cosine values.\n\n    .. code-block:: none\n\n        value = A * cos(w*i + phi) + offset\n\n    Args:\n        w (float) : a frequency for the cosine driver\n        A (float) : an amplitude for the cosine driver\n        phi (float) : a phase offset to start the cosine driver with\n        offset (float) : a global offset to add to the driver values\n\n    '
    from math import cos

    def f(i: float) -> float:
        if False:
            print('Hello World!')
        return A * cos(w * i + phi) + offset
    return partial(force, sequence=_advance(f))

def count() -> partial[Callable[[], None]]:
    if False:
        return 10
    ' Return a driver function that can advance a simple count.\n\n    '
    return partial(force, sequence=_advance(lambda x: x))

def force(f: Callable[[Any], None], sequence: Iterator[Any]) -> Callable[[], None]:
    if False:
        return 10
    ' Return a decorator that can "force" a function with an arbitrary\n    supplied generator\n\n    Args:\n        sequence (iterable) :\n            generator to drive f with\n\n    Returns:\n        decorator\n\n    '

    def wrapper() -> None:
        if False:
            while True:
                i = 10
        f(next(sequence))
    return wrapper

def linear(m: float=1, b: float=0) -> partial[Callable[[], None]]:
    if False:
        i = 10
        return i + 15
    ' Return a driver function that can advance a sequence of linear values.\n\n    .. code-block:: none\n\n        value = m * i + b\n\n    Args:\n        m (float) : a slope for the linear driver\n        x (float) : an offset for the linear driver\n\n    '

    def f(i: float) -> float:
        if False:
            print('Hello World!')
        return m * i + b
    return partial(force, sequence=_advance(f))

def repeat(sequence: Sequence[int]) -> partial[Callable[[], None]]:
    if False:
        print('Hello World!')
    ' Return a driver function that can advance a repeated of values.\n\n    .. code-block:: none\n\n        seq = [0, 1, 2, 3]\n\n        # repeat(seq) => [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, ...]\n\n    Args:\n        sequence (seq) : a sequence of values for the driver to bounce\n\n    '
    N = len(sequence)

    def f(i: int) -> int:
        if False:
            i = 10
            return i + 15
        return sequence[i % N]
    return partial(force, sequence=_advance(f))

def sine(w: float, A: float=1, phi: float=0, offset: float=0) -> partial[Callable[[], None]]:
    if False:
        for i in range(10):
            print('nop')
    ' Return a driver function that can advance a sequence of sine values.\n\n    .. code-block:: none\n\n        value = A * sin(w*i + phi) + offset\n\n    Args:\n        w (float) : a frequency for the sine driver\n        A (float) : an amplitude for the sine driver\n        phi (float) : a phase offset to start the sine driver with\n        offset (float) : a global offset to add to the driver values\n\n    '
    from math import sin

    def f(i: float) -> float:
        if False:
            while True:
                i = 10
        return A * sin(w * i + phi) + offset
    return partial(force, sequence=_advance(f))
T = TypeVar('T')

def _advance(f: Callable[[int], T]) -> Iterable[T]:
    if False:
        return 10
    ' Yield a sequence generated by calling a given function with\n    successively incremented integer values.\n\n    Args:\n        f (callable) :\n            The function to advance\n\n    Yields:\n        f(i) where i increases each call\n\n    '
    i = 0
    while True:
        yield f(i)
        i += 1