import numpy as np
from prml.nn.array.array import Array
from prml.nn.network import Network
from prml.nn.function import Function
from prml.nn.image.util import img2patch, patch2img

class Convolve2dFunction(Function):

    def __init__(self, kernel_size, stride, pad):
        if False:
            i = 10
            return i + 15
        '\n        construct 2 dimensional convolution function\n        Parameters\n        ----------\n        kernel_size : tuple of ints\n            size of convolution kernel\n        stride : tuple of ints\n            stride of kernel application\n        pad : tuple of ints\n            padding image\n        '
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = (0,) + pad + (0,)

    def _forward(self, x, y):
        if False:
            return 10
        img = np.pad(x, [(p,) for p in self.pad], 'constant')
        self.paddedshape = img.shape
        self.patch = img2patch(img, self.kernel_size, self.stride)
        self.outshape = self.patch.shape[:3] + (y.shape[1],)
        self.patch_flattened = self.patch.reshape(-1, y.shape[0])
        return np.matmul(self.patch_flattened, y).reshape(self.outshape)

    def _backward(self, delta, x, y):
        if False:
            i = 10
            return i + 15
        delta_flattened = delta.reshape(-1, delta.shape[-1])
        dpatch_flattened = delta_flattened @ y.T
        dpatch = dpatch_flattened.reshape(self.patch.shape)
        dimg = patch2img(dpatch, self.stride, self.paddedshape)
        slices = tuple((slice(p, len_ - p) for (p, len_) in zip(self.pad, self.paddedshape)))
        dx = dimg[slices]
        dy = self.patch_flattened.T @ delta_flattened
        return (dx, dy)

class Convolve2d(Network):

    def __init__(self, kernel, stride, pad):
        if False:
            while True:
                i = 10
        super().__init__()
        self.in_ch = kernel.shape[-2]
        self.out_ch = kernel.shape[-1]
        self.kernel_size = kernel.shape[:2]
        self.stride = stride
        self.pad = pad
        kernel = kernel.value
        with self.set_parameter():
            self.w = Array(kernel.reshape(-1, kernel.shape[-1]))

    @property
    def kernel(self):
        if False:
            print('Hello World!')
        return self.w.reshape(*self.kernel_size, self.in_ch, self.out_ch)

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        func = Convolve2dFunction(self.kernel_size, self.stride, self.pad)
        return func.forward(x, self.w)

def convolve2d(x, y, stride=(1, 1), pad=(0, 0)):
    if False:
        i = 10
        return i + 15
    "\n    returns convolution of two tensors\n    Parameters\n    ----------\n    x : (n_batch, xlen, ylen, in_chaprml.nnel) Tensor\n        input tensor to be convolved\n    y : (kx, ky, in_chaprml.nnel, out_chaprml.nnel) Tensor\n        convolution kernel\n    stride : tuple of ints (sx, sy)\n        stride of kernel application\n    pad : tuple of ints (px, py)\n        padding image\n    Returns\n    -------\n    output : (n_batch, xlen', ylen', out_chaprml.nnel) Tensor\n        input convolved with kernel\n        len' = (len + 2p - k) // s + 1\n    "
    conv = Convolve2dFunction(y.shape[:2], stride, pad)
    return conv.forward(x, y.reshape(-1, y.shape[-1]))