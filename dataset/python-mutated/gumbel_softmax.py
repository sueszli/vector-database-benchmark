from chainer import backend
import chainer.functions
from chainer import variable

def gumbel_softmax(log_pi, tau=0.1, axis=1):
    if False:
        return 10
    'Gumbel-Softmax sampling function.\n\n    This function draws samples :math:`y_i` from Gumbel-Softmax distribution,\n\n    .. math::\n        y_i = {\\exp((g_i + \\log\\pi_i)/\\tau)\n        \\over \\sum_{j}\\exp((g_j + \\log\\pi_j)/\\tau)},\n\n    where :math:`\\tau` is a temperature parameter and\n    :math:`g_i` s are samples drawn from\n    Gumbel distribution :math:`Gumbel(0, 1)`\n\n    See `Categorical Reparameterization with Gumbel-Softmax\n    <https://arxiv.org/abs/1611.01144>`_.\n\n    Args:\n        log_pi (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable\n            representing pre-normalized log-probability :math:`\\log\\pi`.\n        tau (:class:`~float` or :class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable representing temperature :math:`\\tau`.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    xp = backend.get_array_module(log_pi)
    if log_pi.ndim < 1:
        return variable.Variable(xp.ones((), log_pi.dtype))
    dtype = log_pi.dtype
    g = xp.random.gumbel(size=log_pi.shape).astype(dtype)
    y = chainer.functions.softmax((log_pi + g) / tau, axis=axis)
    return y