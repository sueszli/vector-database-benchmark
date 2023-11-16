import chainer
from chainer import function_node
import chainer.functions
from chainer.utils import precision
from chainer.utils import type_check

class BatchDet(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            i = 10
            return i + 15
        return 'det'

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('x',))
        (a_type,) = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        type_check.expect(a_type.ndim == 3)
        type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    @precision._fp16_mixed_precision_helper
    def forward(self, inputs):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        (x,) = inputs
        xp = chainer.backend.get_array_module(x)
        detx = xp.linalg.det(x)
        return (detx,)

    def backward(self, indexes, gy):
        if False:
            i = 10
            return i + 15
        (x,) = self.get_retained_inputs()
        (detx,) = self.get_retained_outputs()
        (gy,) = gy
        inv_x = chainer.functions.batch_inv(chainer.functions.transpose(x, (0, 2, 1)))
        gy = chainer.functions.broadcast_to(gy[:, None, None], inv_x.shape)
        detx = chainer.functions.broadcast_to(detx[:, None, None], inv_x.shape)
        grad = gy * detx * inv_x
        return (grad,)

def batch_det(a):
    if False:
        while True:
            i = 10
    'Computes the determinant of a batch of square matrices.\n\n    Args:\n        a (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input array to compute the determinant for.\n            The first dimension should iterate over each matrix and be\n            of the batchsize.\n\n    Returns:\n        ~chainer.Variable: vector of determinants for every matrix\n        in the batch.\n\n    '
    return BatchDet().apply((a,))[0]

def det(a):
    if False:
        i = 10
        return i + 15
    'Computes the determinant of a single square matrix.\n\n    Args:\n        a (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input array to compute the determinant for.\n\n    Returns:\n        ~chainer.Variable: Scalar determinant of the matrix a.\n\n    '
    shape = (1, a.shape[0], a.shape[1])
    batched_a = chainer.functions.reshape(a, shape)
    batched_det = BatchDet().apply((batched_a,))[0]
    return chainer.functions.reshape(batched_det, ())