import math
import six
import chainer

def spatial_pyramid_pooling_2d(x, pyramid_height, pooling=None):
    if False:
        print('Hello World!')
    'Spatial pyramid pooling function.\n\n    It outputs a fixed-length vector regardless of input feature map size.\n\n    It performs pooling operation to the input 4D-array ``x`` with different\n    kernel sizes and padding sizes, and then flattens all dimensions except\n    first dimension of all pooling results, and finally concatenates them along\n    second dimension.\n\n    At :math:`i`-th pyramid level, the kernel size\n    :math:`(k_h^{(i)}, k_w^{(i)})` and padding size\n    :math:`(p_h^{(i)}, p_w^{(i)})` of pooling operation are calculated as\n    below:\n\n    .. math::\n        k_h^{(i)} &= \\lceil b_h / 2^i \\rceil, \\\\\n        k_w^{(i)} &= \\lceil b_w / 2^i \\rceil, \\\\\n        p_h^{(i)} &= (2^i k_h^{(i)} - b_h) / 2, \\\\\n        p_w^{(i)} &= (2^i k_w^{(i)} - b_w) / 2,\n\n    where :math:`\\lceil \\cdot \\rceil` denotes the ceiling function, and\n    :math:`b_h, b_w` are height and width of input variable ``x``,\n    respectively. Note that index of pyramid level :math:`i` is zero-based.\n\n    See detail in paper: `Spatial Pyramid Pooling in Deep Convolutional\n    Networks for Visual Recognition\n    <https://arxiv.org/abs/1406.4729>`_.\n\n    Args:\n        x (~chainer.Variable): Input variable. The shape of ``x`` should be\n            ``(batchsize, # of channels, height, width)``.\n        pyramid_height (int): Number of pyramid levels\n        pooling (str):\n            Currently, only ``max`` is supported, which performs a 2d max\n            pooling operation.\n\n    Returns:\n        ~chainer.Variable: Output variable. The shape of the output variable\n        will be :math:`(batchsize, c \\sum_{h=0}^{H-1} 2^{2h}, 1, 1)`,\n        where :math:`c` is the number of channels of input variable ``x``\n        and :math:`H` is the number of pyramid levels.\n    '
    (bottom_c, bottom_h, bottom_w) = x.shape[1:]
    ys = []
    for pyramid_level in six.moves.range(pyramid_height):
        n_bins = int(2 ** pyramid_level)
        ksize_h = int(math.ceil(bottom_h / float(n_bins)))
        remainder_h = ksize_h * n_bins - bottom_h
        pad_h = remainder_h // 2
        ksize_w = int(math.ceil(bottom_w / float(n_bins)))
        remainder_w = ksize_w * n_bins - bottom_w
        pad_w = remainder_w // 2
        ksize = (ksize_h, ksize_w)
        pad = (pad_h, pad_w)
        if pooling != 'max':
            raise ValueError('Unsupported pooling operation: ', pooling)
        y_var = chainer.functions.max_pooling_2d(x, ksize=ksize, stride=None, pad=pad, cover_all=True)
        (n, c, h, w) = y_var.shape
        ys.append(y_var.reshape((n, c * h * w, 1, 1)))
    return chainer.functions.concat(ys)