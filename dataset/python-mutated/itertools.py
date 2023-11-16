"""
Thin wrappers around `itertools`.
"""
import itertools
from ..auto import tqdm as tqdm_auto
__author__ = {'github.com/': ['casperdcl']}
__all__ = ['product']

def product(*iterables, **tqdm_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Equivalent of `itertools.product`.\n\n    Parameters\n    ----------\n    tqdm_class  : [default: tqdm.auto.tqdm].\n    '
    kwargs = tqdm_kwargs.copy()
    tqdm_class = kwargs.pop('tqdm_class', tqdm_auto)
    try:
        lens = list(map(len, iterables))
    except TypeError:
        total = None
    else:
        total = 1
        for i in lens:
            total *= i
        kwargs.setdefault('total', total)
    with tqdm_class(**kwargs) as t:
        it = itertools.product(*iterables)
        for i in it:
            yield i
            t.update()