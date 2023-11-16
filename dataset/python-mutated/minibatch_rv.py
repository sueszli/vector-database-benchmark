from typing import Any, Sequence, Union, cast
import pytensor.tensor as pt
from pytensor import Variable, config
from pytensor.graph import Apply, Op
from pytensor.tensor import NoneConst, TensorVariable, as_tensor_variable
from pymc.logprob.abstract import MeasurableVariable, _logprob, _logprob_helper

class MinibatchRandomVariable(Op):
    """RV whose logprob should be rescaled to match total_size"""
    __props__ = ()
    view_map = {0: [0]}

    def make_node(self, rv, *total_size):
        if False:
            for i in range(10):
                print('nop')
        rv = as_tensor_variable(rv)
        total_size = [as_tensor_variable(t, dtype='int64', ndim=0) if t is not None else NoneConst for t in total_size]
        assert len(total_size) == rv.ndim
        out = rv.type()
        return Apply(self, [rv, *total_size], [out])

    def perform(self, node, inputs, output_storage):
        if False:
            print('Hello World!')
        output_storage[0][0] = inputs[0]
minibatch_rv = MinibatchRandomVariable()
EllipsisType = Any

def create_minibatch_rv(rv: TensorVariable, total_size: Union[int, None, Sequence[Union[int, EllipsisType, None]]]) -> TensorVariable:
    if False:
        return 10
    'Create variable whose logp is rescaled by total_size.'
    if isinstance(total_size, int):
        if rv.ndim <= 1:
            total_size = [total_size]
        else:
            missing_ndims = rv.ndim - 1
            total_size = [total_size] + [None] * missing_ndims
    elif isinstance(total_size, (list, tuple)):
        total_size = list(total_size)
        if Ellipsis in total_size:
            if total_size.count(Ellipsis) > 1:
                raise ValueError('Only one Ellipsis can be present in total_size')
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep + 1:]
            missing_ndims = max((rv.ndim - len(begin) - len(end), 0))
            total_size = begin + [None] * missing_ndims + end
        if len(total_size) > rv.ndim:
            raise ValueError(f'Length of total_size {total_size} is langer than RV ndim {rv.ndim}')
    else:
        raise TypeError(f'Invalid type for total_size: {total_size}')
    return cast(TensorVariable, minibatch_rv(rv, *total_size))

def get_scaling(total_size: Sequence[Variable], shape: TensorVariable) -> TensorVariable:
    if False:
        print('Hello World!')
    'Gets scaling constant for logp.'
    shape = tuple(shape)
    if len(shape) == 0:
        coef = total_size[0] if not NoneConst.equals(total_size[0]) else 1.0
    else:
        coefs = [t / shape[i] for (i, t) in enumerate(total_size) if not NoneConst.equals(t)]
        coef = pt.prod(coefs)
    return pt.cast(coef, dtype=config.floatX)
MeasurableVariable.register(MinibatchRandomVariable)

@_logprob.register(MinibatchRandomVariable)
def minibatch_rv_logprob(op, values, *inputs, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    [value] = values
    (rv, *total_size) = inputs
    return _logprob_helper(rv, value, **kwargs) * get_scaling(total_size, value.shape)