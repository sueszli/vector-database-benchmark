import os
import numpy as np
import jittor as jt
from jittor import nn
f = open(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'src', 'igamma.h'), 'r')
cuda_header = f.read()
f.close()

def igamma(alpha, x):
    if False:
        i = 10
        return i + 15
    cuda_src = '\n        @alias(x, in0)\n        @alias(px ,out0)\n        int batch_size = x_stride0 == 1 ? 1 : x_shape0;\n        int batch_shape = x_shape0 * x_stride0 / batch_size;\n        float alpha = data["alpha"];\n        igamma_kernel<<<batch_size, 16>>>(x_p, px_p, alpha, batch_shape);   \n    '
    out = jt.code(x.shape, x.dtype, [x], cuda_header=cuda_header, cuda_src=cuda_src, data={'alpha': alpha})
    return out