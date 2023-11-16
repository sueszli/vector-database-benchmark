import inspect
import itertools
from . import _funcs_impl, _reductions_impl
from ._normalizations import normalizer

def _public_functions(mod):
    if False:
        while True:
            i = 10

    def is_public_function(f):
        if False:
            return 10
        return inspect.isfunction(f) and (not f.__name__.startswith('_'))
    return inspect.getmembers(mod, is_public_function)
__all__ = []
for (name, func) in itertools.chain(_public_functions(_funcs_impl), _public_functions(_reductions_impl)):
    if name in ['percentile', 'quantile', 'median']:
        decorated = normalizer(func, promote_scalar_result=True)
    elif name == 'einsum':
        decorated = func
    else:
        decorated = normalizer(func)
    decorated.__qualname__ = name
    decorated.__name__ = name
    vars()[name] = decorated
    __all__.append(name)
'\nVendored objects from numpy.lib.index_tricks\n'

class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """

    def __init__(self, maketuple):
        if False:
            while True:
                i = 10
        self.maketuple = maketuple

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if self.maketuple and (not isinstance(item, tuple)):
            return (item,)
        else:
            return item
index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)
__all__ += ['index_exp', 's_']