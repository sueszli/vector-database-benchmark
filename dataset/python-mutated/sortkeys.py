"""Sort key operations."""
from __future__ import annotations
from public import public
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Value
_is_ascending = {'asc': True, 'ascending': True, 'desc': False, 'descending': False, 0: False, 1: True, False: False, True: True}

@public
class SortKey(Value):
    """A sort operation."""
    expr: Value
    ascending: bool = True
    dtype = rlz.dtype_like('expr')
    shape = rlz.shape_like('expr')

    @classmethod
    def __coerce__(cls, value, T=None, S=None):
        if False:
            return 10
        if isinstance(value, tuple):
            (key, asc) = value
        else:
            (key, asc) = (value, True)
        asc = _is_ascending[asc]
        key = super().__coerce__(key, T=T, S=S)
        if isinstance(key, cls):
            return key
        else:
            return cls(key, asc)

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self.expr.name

    @property
    def descending(self) -> bool:
        if False:
            while True:
                i = 10
        return not self.ascending