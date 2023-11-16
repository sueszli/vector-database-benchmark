import ivy
import ivy.functional.frontends.jax as jax_frontend
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back

class _IndexUpdateHelper:
    __slots__ = ('array',)

    def __init__(self, array):
        if False:
            print('Hello World!')
        self.array = array

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return _IndexUpdateRef(self.array, index)

    def __setitem__(self, index):
        if False:
            i = 10
            return i + 15
        return _IndexUpdateRef(self.array, index)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'_IndexUpdateHelper({repr(self.array)})'

class _IndexUpdateRef:
    __slots__ = ('array', 'index')

    def __init__(self, array, index):
        if False:
            print('Hello World!')
        self.array = array
        self.index = index

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'_IndexUpdateRef({repr(self.array)}, {repr(self.index)})'

    def get(self, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None):
        if False:
            i = 10
            return i + 15
        return _rewriting_take(self.array, self.index, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode=mode, fill_value=fill_value)

    def set(self, values, indices_are_sorted=False, unique_indices=False, mode=None):
        if False:
            for i in range(10):
                print('nop')
        ret = ivy.copy_array(self.array)
        if hasattr(values, 'ivy_array'):
            ret[self.index] = values.ivy_array
        else:
            ret[self.index] = values
        return jax_frontend.Array(ret)

@to_ivy_arrays_and_back
def _rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False, mode=None, fill_value=None):
    if False:
        for i in range(10):
            print('nop')
    return ivy.get_item(arr, idx)