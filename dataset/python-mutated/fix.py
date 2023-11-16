import chainer
from chainer import backend
from chainer import utils

def fix(x):
    if False:
        while True:
            i = 10
    'Elementwise fix function.\n\n    .. math::\n       y_i = \\lfix x_i \\rfix\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    if isinstance(x, chainer.variable.Variable):
        x = x.array
    xp = backend.get_array_module(x)
    return chainer.as_variable(utils.force_array(xp.fix(x), x.dtype))