import numpy
import six
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class EmbedIDFunction(function_node.FunctionNode):

    def __init__(self, ignore_label=None):
        if False:
            i = 10
            return i + 15
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check.expect(in_types.size() == 2)
        (x_type, w_type) = in_types
        type_check.expect(x_type.dtype.kind == 'i', x_type.ndim >= 1)
        type_check.expect(w_type.dtype.kind == 'f', w_type.ndim == 2)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((0,))
        (x, W) = inputs
        self._w_shape = W.shape
        xp = backend.get_array_module(*inputs)
        if chainer.is_debug():
            valid_x = xp.logical_and(0 <= x, x < len(W))
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy `0 <= x < len(W)`')
        if self.ignore_label is not None:
            mask = x == self.ignore_label
            return (xp.where(mask[..., None], 0, W[xp.where(mask, 0, x)]),)
        return (W[x],)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        inputs = self.get_retained_inputs()
        gW = EmbedIDGrad(self._w_shape, self.ignore_label).apply(inputs + grad_outputs)[0]
        return (None, gW)

class EmbedIDGrad(function_node.FunctionNode):

    def __init__(self, w_shape, ignore_label=None):
        if False:
            i = 10
            return i + 15
        self.w_shape = w_shape
        self.ignore_label = ignore_label

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.retain_inputs((0,))
        xp = backend.get_array_module(*inputs)
        (x, gy) = inputs
        self._gy_shape = gy.shape
        gW = xp.zeros(self.w_shape, dtype=gy.dtype)
        if xp is numpy:
            for (ix, igy) in six.moves.zip(x.ravel(), gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            utils.nondeterministic('atomicAdd')
            if self.ignore_label is None:
                cuda.elementwise('T gy, S x, S n_out', 'raw T gW', 'ptrdiff_t w_ind[] = {x, i % n_out};atomicAdd(&gW[w_ind], gy)', 'embed_id_bwd')(gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise('T gy, S x, S n_out, S ignore', 'raw T gW', '\n                    if (x != ignore) {\n                      ptrdiff_t w_ind[] = {x, i % n_out};\n                      atomicAdd(&gW[w_ind], gy);\n                    }\n                    ', 'embed_id_bwd_ignore_label')(gy, xp.expand_dims(x, -1), gW.shape[1], self.ignore_label, gW)
        return (gW,)

    def backward(self, indexes, grads):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*grads)
        x = self.get_retained_inputs()[0].data
        ggW = grads[0]
        if self.ignore_label is not None:
            mask = x == self.ignore_label
            if not 0 <= self.ignore_label < self.w_shape[1]:
                x = xp.where(mask, 0, x)
        ggy = ggW[x]
        if self.ignore_label is not None:
            (mask, zero, _) = xp.broadcast_arrays(mask[..., None], xp.zeros((), ggy.dtype), ggy.data)
            ggy = chainer.functions.where(mask, zero, ggy)
        return (None, ggy)

def embed_id(x, W, ignore_label=None):
    if False:
        i = 10
        return i + 15
    'Efficient linear function for one-hot input.\n\n    This function implements so called *word embeddings*. It takes two\n    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer\n    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`\n    float matrix. It outputs :math:`B \\times d` matrix whose ``i``-th\n    row is the ``x[i]``-th row of ``W``.\n\n    This function is only differentiable on the input ``W``.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Batch vectors of IDs. Each element must be signed integer.\n        W (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Distributed representation of each ID (a.k.a. word embeddings).\n        ignore_label (:class:`int` or :class:`None`):\n            If ``ignore_label`` is an int value, ``i``-th row of return\n            value is filled with ``0``.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    .. seealso::\n\n        :class:`~chainer.links.EmbedID` to manage the model parameter ``W``.\n\n    .. admonition:: Example\n\n        >>> x = np.array([2, 1]).astype(np.int32)\n        >>> x\n        array([2, 1], dtype=int32)\n        >>> W = np.array([[0, 0, 0],\n        ...               [1, 1, 1],\n        ...               [2, 2, 2]]).astype(np.float32)\n        >>> W\n        array([[0., 0., 0.],\n               [1., 1., 1.],\n               [2., 2., 2.]], dtype=float32)\n        >>> F.embed_id(x, W).array\n        array([[2., 2., 2.],\n               [1., 1., 1.]], dtype=float32)\n        >>> F.embed_id(x, W, ignore_label=1).array\n        array([[2., 2., 2.],\n               [0., 0., 0.]], dtype=float32)\n\n    '
    return EmbedIDFunction(ignore_label=ignore_label).apply((x, W))[0]