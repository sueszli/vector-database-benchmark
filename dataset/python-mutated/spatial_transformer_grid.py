import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.libcudnn
    _sampler_type = cuda.libcudnn.CUDNN_SAMPLER_BILINEAR

class SpatialTransformerGrid(function.Function):

    def __init__(self, output_shape):
        if False:
            return 10
        self.output_shape = output_shape

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('theta',))
        theta_type = in_types[0]
        type_check.expect(theta_type.dtype.kind == 'f', theta_type.ndim == 3, theta_type.shape[1] == 2, theta_type.shape[2] == 3)

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        return self._forward(inputs)

    def forward_gpu(self, inputs):
        if False:
            print('Hello World!')
        if not chainer.should_use_cudnn('>=auto', 5000):
            return self._forward(inputs)
        (theta,) = inputs
        (B, _, _) = theta.shape
        (H, W) = self.output_shape
        grid_t = cuda.cupy.empty((B, H, W, 2), dtype=theta.dtype)
        shape = numpy.array((B, 1, H, W), dtype=numpy.int32)
        theta = cuda.cupy.ascontiguousarray(theta)
        handle = cudnn.get_handle()
        self.st_desc = cuda.cupy.cudnn.create_spatial_transformer_descriptor(_sampler_type, grid_t.dtype, len(shape), shape.ctypes.data)
        libcudnn.spatialTfGridGeneratorForward(handle, self.st_desc.value, theta.data.ptr, grid_t.data.ptr)
        grid = cuda.cupy.transpose(grid_t, (0, 3, 1, 2))
        return (grid,)

    def _forward(self, inputs):
        if False:
            i = 10
            return i + 15
        (theta,) = inputs
        (H, W) = self.output_shape
        (B, _, _) = theta.shape
        xp = backend.get_array_module(theta)
        (ys, xs) = xp.meshgrid(xp.linspace(-1, 1, H, dtype=theta.dtype), xp.linspace(-1, 1, W, dtype=theta.dtype), indexing='ij', copy=False)
        coords = xp.concatenate([xs[None], ys[None], xp.ones((1, H, W), dtype=theta.dtype)], axis=0)
        grid = theta.dot(coords.reshape(3, H * W)).reshape(B, 2, H, W)
        return (grid,)

    def backward_cpu(self, inputs, grad_outputs):
        if False:
            return 10
        return self._backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        if not chainer.should_use_cudnn('>=auto', 5000):
            return self._backward(inputs, grad_outputs)
        (theta,) = inputs
        (ggrid,) = grad_outputs
        ggrid_t = cuda.cupy.transpose(ggrid, (0, 2, 3, 1))
        gtheta = cuda.cupy.empty_like(theta)
        handle = cudnn.get_handle()
        ggrid_t = cuda.cupy.ascontiguousarray(ggrid_t)
        libcudnn.spatialTfGridGeneratorBackward(handle, self.st_desc.value, ggrid_t.data.ptr, gtheta.data.ptr)
        return (gtheta,)

    def _backward(self, inputs, grad_outputs):
        if False:
            while True:
                i = 10
        (theta,) = inputs
        (ggrid,) = grad_outputs
        (H, W) = self.output_shape
        (B, _, _) = theta.shape
        xp = backend.get_array_module(theta)
        (ys, xs) = xp.meshgrid(xp.linspace(-1, 1, H, dtype=theta.dtype), xp.linspace(-1, 1, W, dtype=theta.dtype), indexing='ij', copy=False)
        coords = xp.concatenate([xs[None], ys[None], xp.ones((1, H, W), dtype=theta.dtype)], axis=0)
        coords_T = coords.reshape(3, H * W).transpose(1, 0)
        ggrid = ggrid.reshape(B, 2, H * W)
        gtheta = ggrid.dot(coords_T).reshape(B, 2, 3)
        return (gtheta,)

def spatial_transformer_grid(theta, output_shape, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '2D Spatial Transformer grid.\n\n    This function generates coordinates of the points sampled from an image\n    to perform warping described in `Spatial Transformer Networks\n    <https://arxiv.org/abs/1506.02025>`_.\n\n    Given a coordinate in the warped image :math:`(x_i^t, y_i^t)`, the point\n    sampled from the source image :math:`(x_i^s, y_i^s)` are calculated\n    by the following equation.\n\n    .. note::\n\n        cuDNN supports SpatialTransformerGrid from version 5.0.0.\n\n    .. math::\n\n        \\left(\\begin{matrix} x_i^s \\\\\n            y_i^s \\end{matrix}\\right)\n        =\n        \\left(\\begin{matrix} \\theta_{11} & \\theta_{12} & \\theta_{13} \\\\\n            \\theta_{21} & \\theta_{22} & \\theta_{23} \\end{matrix}\\right)\n        \\left(\\begin{matrix} x_i^t \\\\\n            y_i^t \\\\\n            1 \\end{matrix}\\right)\n\n    Notation: here is a notation for dimensionalities.\n\n    - :math:`n` is the batch size.\n    - :math:`h_O` and :math:`w_O` are the height and the width of the output\n      image.\n\n    Args:\n        theta (:class:`~chainer.Variable` or :ref:`ndarray`):\n            An array of shape :math:`(n, 2, 3)`.\n            This is a batch of :math:`2 \\times 3` matrix used for\n            the warping described above.\n        output_shape (tuple): A tuple of 2 elements: :math:`h_O, w_O`.\n\n    Returns:\n        ~chainer.Variable:  A variable of shape :math:`(n, 2, h_O, w_O)`.\n        In the 2nd dimension, the first element is the coordinate along the\n        x axis, and the second element is the coordinate along the y axis.\n        All the coordinates in the image are scaled to fit range\n        :math:`[-1, 1]`.\n        This means that the coordinate :math:`(-1, -1)` corresponds to\n        the upper-left corner of the input image.\n\n    '
    if kwargs:
        argument.check_unexpected_kwargs(kwargs, use_cudnn='The argument "use_cudnn" is not supported anymore. Use chainer.using_config(\'use_cudnn\', value) context where value can be `always`, `never`, or `auto`.')
        argument.assert_kwargs_empty(kwargs)
    return SpatialTransformerGrid(output_shape)(theta)