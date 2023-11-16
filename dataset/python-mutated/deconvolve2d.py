import numpy as np
from prml.nn.array.array import Array
from prml.nn.function import Function
from prml.nn.network import Network
from prml.nn.image.util import img2patch, patch2img

class Deconvolve2dFunction(Function):

    def __init__(self, kernel_size, out_ch, stride, pad, shape):
        if False:
            while True:
                i = 10
        '\n        construct 2 dimensional convolution function\n        Parameters\n        ----------\n        stride : tuple of ints\n            stride of kernel application\n        pad : tuple of ints\n            padding image\n        shape : tuple of ints, optional\n            desired output image shape\n        '
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.stride = stride
        self.pad = (0,) + pad + (0,)
        if shape is None:
            self.shape = None
        else:
            self.shape = shape

    def _forward(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        if self.shape is None:
            shape = (len(x),) + tuple((s * (imlen - 1) + klen for (s, imlen, klen) in zip(self.stride, x.shape[1:], self.kernel_size))) + (self.out_ch,)
        else:
            shape = (len(x),) + self.shape + (self.out_ch,)
        patch_flat = np.matmul(x, y.T)
        output = patch2img(patch_flat.reshape(*patch_flat.shape[:3], *self.kernel_size, -1), self.stride, shape)
        output = output[:, self.pad[1]:output.shape[1] - self.pad[1], self.pad[2]:output.shape[2] - self.pad[2]]
        return output

    def _backward(self, delta, x, y):
        if False:
            return 10
        delta = np.pad(delta, [(p,) for p in self.pad], 'constant')
        dpatch = img2patch(delta, self.kernel_size, self.stride)
        dpatch_flat = dpatch.reshape(-1, y.shape[0])
        dx = np.matmul(dpatch_flat, y).reshape(x.shape)
        dy = np.matmul(x.reshape(-1, x.shape[-1]).T, dpatch_flat).T
        return (dx, dy)

class Deconvolve2d(Network):

    def __init__(self, kernel, stride, pad, shape=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.kernel_size = kernel.shape[:2]
        self.out_ch = kernel.shape[2]
        self.in_ch = kernel.shape[3]
        self.stride = stride
        self.pad = pad
        self.shape = shape
        kernel = kernel.value
        with self.set_parameter():
            self.w = Array(kernel.reshape(-1, kernel.shape[-1]))

    @property
    def kernel(self):
        if False:
            while True:
                i = 10
        return self.w.reshape(*self.kernel_size, self.out_ch, self.in_ch)

    def __call__(self, x):
        if False:
            while True:
                i = 10
        func = Deconvolve2dFunction(self.kernel_size, self.out_ch, self.stride, self.pad, self.shape)
        return func.forward(x, self.w)

def deconvolve2d(x, y, stride=1, pad=0, shape=None):
    if False:
        return 10
    "\n    deconvolution of two tensors\n    aka transposed convolution\n\n    Parameters\n    ----------\n    x : (n_batch, xlen, ylen, in_chaprml.nnel) Tensor\n        input tensor to be deconvolved\n    y : (kx, ky, out_chaprml.nnel, in_chaprml.nnel) Tensor\n        deconvolution kernel\n    stride : int or tuple of ints (sx, sy)\n        stride of kernel application\n    pad : int or tuple of ints (px, py)\n        padding image\n    shape : tuple of ints (xlen', ylen')\n        desired shape of output image\n        If not specified, the output has the following length\n        len' = s * (len - 1) - 2p + k\n\n    Returns\n    -------\n    output : (n_batch, xlen', ylen', out_chaprml.nnel) Tensor\n        The first argument deconvolved with the second one\n        len' will be the following if not specified\n        len' = s * (len - 1) - 2p + k\n    "
    deconv = Deconvolve2dFunction(y.shape[:2], y.shape[2], stride, pad, shape)
    return deconv.forward(x, y.reshape(-1, y.shape[-1]))