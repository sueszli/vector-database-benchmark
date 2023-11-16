from __future__ import annotations
import sys
from ._utils.hacks import not_iterable

def _arithm_op(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    import nvidia.dali.ops
    setattr(sys.modules[__name__], '_arithm_op', nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs)

class _NewAxis:

    def __init__(self, name=None):
        if False:
            print('Hello World!')
        if name is not None:
            if not isinstance(name, str):
                raise TypeError('Axis name must be a single-character string')
            if len(name) != 1:
                raise ValueError('Axis name must be a single-character string')
        self._name = name

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def __call__(self, name=None):
        if False:
            while True:
                i = 10
        return _NewAxis(name)
newaxis = _NewAxis()

class DataNode(object):
    """This class is a symbolic representation of a TensorList and is used at graph definition
    stage. It does not carry actual data, but is used to define the connections between operators
    and to specify the pipeline outputs. See documentation for :class:`Pipeline` for details.

    ``DataNode`` objects can be passed to DALI operators as inputs (and some of the named keyword
    arguments) but they also provide arithmetic operations which implicitly create appropriate
    operators that perform the expressions.
    """

    def __init__(self, name, device='cpu', source=None):
        if False:
            return 10
        self.name = name
        self.device = device
        self.source = source

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'DataNode(name="{self.name}", device="{self.device}")'
    __repr__ = __str__

    def gpu(self) -> DataNode:
        if False:
            print('Hello World!')
        from nvidia.dali import _conditionals
        if _conditionals.conditionals_enabled():
            ([self_split], _) = _conditionals.apply_conditional_split_to_args([self], {})
            transferred_node = DataNode(self_split.name, 'gpu', self_split.source)
            _conditionals.register_data_nodes(transferred_node, [self])
            return transferred_node
        return DataNode(self.name, 'gpu', self.source)

    def __add__(self, other) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return _arithm_op('add', self, other)

    def __radd__(self, other) -> DataNode:
        if False:
            while True:
                i = 10
        return _arithm_op('add', other, self)

    def __sub__(self, other) -> DataNode:
        if False:
            print('Hello World!')
        return _arithm_op('sub', self, other)

    def __rsub__(self, other) -> DataNode:
        if False:
            i = 10
            return i + 15
        return _arithm_op('sub', other, self)

    def __mul__(self, other) -> DataNode:
        if False:
            while True:
                i = 10
        return _arithm_op('mul', self, other)

    def __rmul__(self, other) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return _arithm_op('mul', other, self)

    def __pow__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('pow', self, other)

    def __rpow__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('pow', other, self)

    def __truediv__(self, other) -> DataNode:
        if False:
            print('Hello World!')
        return _arithm_op('fdiv', self, other)

    def __rtruediv__(self, other) -> DataNode:
        if False:
            i = 10
            return i + 15
        return _arithm_op('fdiv', other, self)

    def __floordiv__(self, other) -> DataNode:
        if False:
            while True:
                i = 10
        return _arithm_op('div', self, other)

    def __rfloordiv__(self, other) -> DataNode:
        if False:
            i = 10
            return i + 15
        return _arithm_op('div', other, self)

    def __neg__(self) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return _arithm_op('minus', self)

    def __pos__(self) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return self

    def __eq__(self, other) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return _arithm_op('eq', self, other)

    def __ne__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('neq', self, other)

    def __lt__(self, other) -> DataNode:
        if False:
            print('Hello World!')
        return _arithm_op('lt', self, other)

    def __le__(self, other) -> DataNode:
        if False:
            print('Hello World!')
        return _arithm_op('leq', self, other)

    def __gt__(self, other) -> DataNode:
        if False:
            i = 10
            return i + 15
        return _arithm_op('gt', self, other)

    def __ge__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('geq', self, other)

    def __and__(self, other) -> DataNode:
        if False:
            while True:
                i = 10
        return _arithm_op('bitand', self, other)

    def __rand__(self, other) -> DataNode:
        if False:
            i = 10
            return i + 15
        return _arithm_op('bitand', other, self)

    def __or__(self, other) -> DataNode:
        if False:
            print('Hello World!')
        return _arithm_op('bitor', self, other)

    def __ror__(self, other) -> DataNode:
        if False:
            for i in range(10):
                print('nop')
        return _arithm_op('bitor', other, self)

    def __xor__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('bitxor', self, other)

    def __rxor__(self, other) -> DataNode:
        if False:
            return 10
        return _arithm_op('bitxor', other, self)

    def __bool__(self):
        if False:
            return 10
        raise TypeError('"DataNode" was used in conditional context - it might have been used in truth evaluation for `if` statement, logical expression or cast to a boolean. To use conditional execution via `if` statements you need to specify `enable_conditionals=True` in `@nvidia.dali.pipeline_def` decorator. You can read more about conditional execution in specific section of the Pipeline documentation. Bool conversion can be achieved with the `cast` operator.')

    def __getitem__(self, val) -> DataNode:
        if False:
            while True:
                i = 10
        idxs = []
        new_axes = []
        new_axis_names = []

        def process_index(idx, dim):
            if False:
                while True:
                    i = 10
            if idx is None:
                idxs.append((None, None, None, None))
                return True
            elif isinstance(idx, slice):
                idxs.append((None, idx.start, idx.stop, idx.step))
                return True
            if isinstance(idx, _NewAxis):
                new_axes.append(dim)
                if idx.name is not None:
                    new_axis_names.append(idx.name)
                return True
            if idx is Ellipsis:
                raise NotImplementedError('Ellipsis in indexing is not implemented')
            if isinstance(idx, (float, str)):
                raise TypeError('Invalid type for an index: ', type)
            idxs.append((idx, None, None, None))
            return False
        if not isinstance(val, tuple):
            val = (val,)
        d = 0
        for v in val:
            if process_index(v, d):
                d += 1
        if len(new_axis_names) != 0:
            if len(new_axis_names) != len(new_axes):
                raise ValueError('New axis name must be specified for all axes or none.')
            new_axis_names = ''.join(new_axis_names)
        else:
            new_axis_names = None
        slice_args = {}
        for (i, (at, lo, hi, step)) in enumerate(idxs):
            if at is not None:
                slice_args['at_%i' % i] = at
            if lo is not None:
                slice_args['lo_%i' % i] = lo
            if hi is not None:
                slice_args['hi_%i' % i] = hi
            if step is not None:
                slice_args['step_%i' % i] = step
        import nvidia.dali.fn
        if len(slice_args) == 0:
            if len(new_axes) > 0 and isinstance(val[-1], _NewAxis):
                sliced = self
            else:
                sliced = nvidia.dali.fn.subscript_dim_check(self, num_subscripts=len(idxs))
        else:
            sliced = nvidia.dali.fn.tensor_subscript(self, **slice_args, num_subscripts=len(idxs))
        if len(new_axes) == 0:
            return sliced
        else:
            return nvidia.dali.fn.expand_dims(sliced, axes=new_axes, new_axis_names=new_axis_names)
not_iterable(DataNode)

def _check(maybe_node):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(maybe_node, DataNode):
        raise TypeError(f'Expected outputs of type compatible with "DataNode". Received output type with name "{type(maybe_node).__name__}" that does not match.')