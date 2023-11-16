import six
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

class TheanoFunction(function.Function):

    def __init__(self, forward_func, backward_func):
        if False:
            while True:
                i = 10
        self.forward_func = forward_func
        self.backward_func = backward_func

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check.expect(in_types.size() == len(self.forward_func.indices))
        for (actual_type, input_info) in six.moves.zip(in_types, self.forward_func.indices):
            expect_type = input_info[0].variable.type
            type_check.expect(actual_type.ndim == expect_type.ndim, actual_type.dtype == expect_type.numpy_dtype)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        gpu = backend.get_array_module(*inputs) is cuda.cupy
        inputs = [cuda.to_cpu(x) for x in inputs]
        outputs = self.forward_func(*inputs)
        if gpu:
            device = cuda.get_device_from_array(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]
        return tuple(outputs)

    def backward(self, inputs, grads):
        if False:
            print('Hello World!')
        gpu = backend.get_array_module(*inputs) is cuda.cupy
        args = [cuda.to_cpu(x) for x in inputs + grads]
        outputs = self.backward_func(*args)
        assert len(outputs) == len(inputs)
        if gpu:
            device = cuda.get_device_from_array(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]
        results = []
        for (o, i) in zip(outputs, inputs):
            if i.dtype.kind != 'f':
                o = None
            elif o.dtype != i.dtype:
                o = o.astype(i.dtype)
            results.append(o)
        return tuple(results)

def theano_function(forward_func, backward_func, *inputs):
    if False:
        i = 10
        return i + 15
    return TheanoFunction(forward_func, backward_func)(*inputs)