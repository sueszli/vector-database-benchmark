from typing import List, Sequence, Union, Callable
from nvidia.dali.data_node import DataNode as _DataNode

class _CompoundOp:

    def __init__(self, op_list):
        if False:
            return 10
        self._ops = []
        for op in op_list:
            if isinstance(op, _CompoundOp):
                self._ops += op._ops
            else:
                self._ops.append(op)

    def __call__(self, *inputs: _DataNode, **kwargs) -> Union[Sequence[_DataNode], _DataNode, None]:
        if False:
            print('Hello World!')
        inputs = list(inputs)
        for op in self._ops:
            for i in range(len(inputs)):
                if inputs[i].device == 'cpu' and op.device == 'gpu' and (op.schema.GetInputDevice(i) != 'cpu'):
                    inputs[i] = inputs[i].gpu()
            inputs = op(*inputs, **kwargs)
            kwargs = {}
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            if isinstance(inputs, _DataNode):
                inputs = [inputs]
        return inputs[0] if len(inputs) == 1 else inputs

def Compose(op_list: List[Callable[..., Union[Sequence[_DataNode], _DataNode]]]) -> _CompoundOp:
    if False:
        i = 10
        return i + 15
    'Returns a meta-operator that chains the operations in op_list.\n\nThe return value is a callable object which, when called, performs::\n\n    op_list[n-1](op_list([n-2](...  op_list[0](args))))\n\nOperators can be composed only when all outputs of the previous operator can be processed directly\nby the next operator in the list.\n\nThe example below chains an image decoder and a Resize operation with random square size.\nThe  ``decode_and_resize`` object can be called as if it was an operator::\n\n    decode_and_resize = ops.Compose([\n        ops.decoders.Image(device="cpu"),\n        ops.Resize(size=fn.random.uniform(range=400,500)), device="gpu")\n    ])\n\n    files, labels = fn.readers.caffe(path=caffe_db_folder, seed=1)\n    pipe.set_ouputs(decode_and_resize(files), labels)\n\nIf there\'s a transition from CPU to GPU in the middle of the ``op_list``, as is the case in this\nexample, ``Compose`` automatically arranges copying the data to GPU memory.\n\n\n.. note::\n    This is an experimental feature, subject to change without notice.\n'
    return op_list[0] if len(op_list) == 1 else _CompoundOp(op_list)