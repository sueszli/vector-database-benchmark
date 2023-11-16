from chainer.functions.activation import elu

def selu(x, alpha=1.6732632423543772, scale=1.0507009873554805):
    if False:
        while True:
            i = 10
    'Scaled Exponential Linear Unit function.\n\n    For parameters :math:`\\alpha` and :math:`\\lambda`, it is expressed as\n\n    .. math::\n        f(x) = \\lambda \\left \\{ \\begin{array}{ll}\n        x & {\\rm if}~ x \\ge 0 \\\\\n        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,\n        \\end{array} \\right.\n\n    See: https://arxiv.org/abs/1706.02515\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n        alpha (float): Parameter :math:`\\alpha`.\n        scale (float): Parameter :math:`\\lambda`.\n\n    Returns:\n        ~chainer.Variable: Output variable. A\n        :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n\n    '
    return scale * elu.elu(x, alpha=alpha)