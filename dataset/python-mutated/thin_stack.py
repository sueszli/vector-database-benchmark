import chainer
from chainer import backend
from chainer.utils import type_check

class ThinStackSet(chainer.Function):
    """Set values to a thin stack."""

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check.expect(in_types.size() == 3)
        (s_type, i_type, v_type) = in_types
        type_check.expect(s_type.dtype.kind == 'f', i_type.dtype.kind == 'i', s_type.dtype == v_type.dtype, s_type.ndim == 3, i_type.ndim == 1, v_type.ndim == 2, s_type.shape[0] >= i_type.shape[0], i_type.shape[0] == v_type.shape[0], s_type.shape[2] == v_type.shape[1])

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        (stack, indices, values) = inputs
        stack[xp.arange(len(indices)), indices] = values
        return (stack,)

    def backward(self, inputs, grads):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(*inputs)
        (_, indices, _) = inputs
        g = grads[0]
        gv = g[xp.arange(len(indices)), indices]
        g[xp.arange(len(indices)), indices] = 0
        return (g, None, gv)

def thin_stack_set(s, i, x):
    if False:
        return 10
    return ThinStackSet()(s, i, x)

class ThinStackGet(chainer.Function):

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check.expect(in_types.size() == 2)
        (s_type, i_type) = in_types
        type_check.expect(s_type.dtype.kind == 'f', i_type.dtype.kind == 'i', s_type.ndim == 3, i_type.ndim == 1, s_type.shape[0] >= i_type.shape[0])

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*inputs)
        (stack, indices) = inputs
        return (stack[xp.arange(len(indices)), indices], stack)

    def backward(self, inputs, grads):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        (stack, indices) = inputs
        (g, gs) = grads
        if gs is None:
            gs = xp.zeros_like(stack)
        if g is not None:
            gs[xp.arange(len(indices)), indices] += g
        return (gs, None)

def thin_stack_get(s, i):
    if False:
        print('Hello World!')
    return ThinStackGet()(s, i)