import numpy
from chainer import backend
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

def _hinge_fwd_kernel():
    if False:
        print('Hello World!')
    return cuda.elementwise('S t', 'raw T bottom_diff', 'int ind[] = {i, t}; bottom_diff[ind] *= -1', 'hinge_fwd')

class Hinge(function.Function):
    """Hinge loss."""

    def __init__(self, norm='L1', reduce='mean'):
        if False:
            for i in range(10):
                print('nop')
        if norm in ['L1', 'L2']:
            self.norm = norm
        else:
            raise NotImplementedError("norm should be either 'L1' or 'L2'")
        if reduce in ['mean', 'no']:
            self.reduce = reduce
        else:
            raise ValueError("only 'mean' and 'no' are valid for 'reduce', but '%s' is given" % reduce)

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x', 't'))
        (x_type, t_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', t_type.dtype.kind == 'i', x_type.ndim == 2, t_type.ndim == 1, x_type.shape[0] == t_type.shape[0])

    def forward_cpu(self, inputs):
        if False:
            print('Hello World!')
        (x, t) = inputs
        num = len(x)
        self.bottom_diff = numpy.copy(x)
        self.bottom_diff[numpy.arange(num), t] *= -1
        self.bottom_diff = numpy.maximum(0, 1 + self.bottom_diff)
        if self.norm == 'L1':
            loss = self.bottom_diff
        elif self.norm == 'L2':
            loss = self.bottom_diff ** 2
        else:
            raise NotImplementedError()
        if self.reduce == 'mean':
            loss = loss.sum() / num
        return (numpy.array(loss, dtype=x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            return 10
        (x, t) = inputs
        num = x.dtype.type(len(x))
        self.bottom_diff = cuda.cupy.maximum(0, 1 + _hinge_fwd_kernel()(t, x.copy()))
        if self.norm == 'L1':
            loss = self.bottom_diff
        elif self.norm == 'L2':
            loss = self.bottom_diff ** 2
        else:
            raise NotImplementedError()
        if self.reduce == 'mean':
            loss = loss.sum() / num
        return (loss,)

    def backward_cpu(self, inputs, grad_outputs):
        if False:
            while True:
                i = 10
        (t, gloss) = (inputs[1], grad_outputs[0])
        if self.reduce == 'mean':
            gloss /= len(t)
        self.bottom_diff[numpy.arange(len(t)), t] *= -1
        if self.norm == 'L1':
            gx = gloss * numpy.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = 2 * gloss * self.bottom_diff
        else:
            raise NotImplementedError()
        return (gx, None)

    def backward_gpu(self, inputs, grad_outputs):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(*inputs)
        (t, gloss) = (inputs[1], grad_outputs[0])
        if self.reduce == 'mean':
            gloss /= len(t)
        self.bottom_diff = _hinge_fwd_kernel()(t, self.bottom_diff)
        if self.norm == 'L1':
            gx = gloss * xp.sign(self.bottom_diff)
        elif self.norm == 'L2':
            gx = 2 * gloss * self.bottom_diff
        else:
            raise NotImplementedError()
        return (gx, None)

def hinge(x, t, norm='L1', reduce='mean'):
    if False:
        return 10
    "Computes the hinge loss for a one-of-many classification task.\n\n        .. math::\n            L = \\frac{1}{N} \\sum_{n=1}^N \\sum_{k=1}^K \\left[\n            \\max(0, 1 - \\delta\\{t_n = k\\} x_{nk}) \\right]^p\n\n        where :math:`N` denotes the batch size and :math:`K` is the number of\n        classes of interest,\n\n        .. math::\n            \\delta \\{ {\\rm condition} \\} = \\left \\{ \\begin{array}{cc}\n            1 & {\\rm if~condition\\ is\\ true} \\\\\n            -1 & {\\rm otherwise,}\n            \\end{array} \\right.\n\n        and\n\n        .. math::\n            p = \\left \\{ \\begin{array}{cc}\n            1 & {\\rm if~norm} = {\\rm L1} \\\\\n            2 & {\\rm if~norm} = {\\rm L2.}\n            \\end{array} \\right.\n\n        Let the hinge loss function :math:`l(x, \\delta)` be\n        :math:`\\left[\\max(0, 1 - \\delta x) \\right]^p`.\n        When :math:`x` and :math:`\\delta` have the same sign (meaning\n        :math:`x` predicts the proper score for classification) and\n        :math:`|x| \\geq 1`, the hinge loss :math:`l(x, \\delta) = 0`, but when\n        they have opposite sign, :math:`l(x, \\delta)` increases linearly\n        with :math:`x`.\n\n        The output is a variable whose value depends on the value of\n        the option ``reduce``. If it is ``'no'``, it holds the elementwise\n        loss values. If it is ``'mean'``, it takes the mean of loss values.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n            The shape of ``x`` should be (:math:`N`, :math:`K`).\n        t (:class:`~chainer.Variable` or :ref:`ndarray`):\n            The :math:`N`-dimensional label vector with values\n            :math:`t_n \\in \\{0, 1, 2, \\dots, K-1\\}`.\n            The shape of ``t`` should be (:math:`N`,).\n        norm (string): Specifies norm type. Either ``'L1'`` or ``'L2'`` is\n            acceptable.\n        reduce (str): Reduction option. Its value must be either\n            ``'mean'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.\n\n    Returns:\n        ~chainer.Variable:\n            A variable object holding a scalar array of the\n            hinge loss :math:`L`.\n            If ``reduce`` is ``'no'``, the output variable holds array\n            whose shape is same as one of (hence both of) input variables.\n            If it is ``'mean'``, the output variable holds a scalar value.\n\n    .. admonition:: Example\n\n        In this case, the batch size ``N`` is 2 and the number of classes ``K``\n        is 3.\n\n        >>> x = np.array([[-2.0, 3.0, 0.5],\n        ...               [5.0, 2.0, -0.5]]).astype(np.float32)\n        >>> x\n        array([[-2. ,  3. ,  0.5],\n               [ 5. ,  2. , -0.5]], dtype=float32)\n        >>> t = np.array([1, 0]).astype(np.int32)\n        >>> t\n        array([1, 0], dtype=int32)\n        >>> F.hinge(x, t)\n        variable(2.5)\n        >>> F.hinge(x, t, reduce='no')\n        variable([[0. , 0. , 1.5],\n                  [0. , 3. , 0.5]])\n        >>> F.hinge(x, t, norm='L2')\n        variable(5.75)\n\n    "
    return Hinge(norm, reduce)(x, t)