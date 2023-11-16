from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import expand_dims
from chainer.functions.array import reshape
from chainer.functions.connection import embed_id
from chainer.functions.math import average
from chainer.functions.math import exponential
from chainer.functions.math import logsumexp
from chainer.functions.math import matmul
from chainer.functions.math import sum as _sum

def black_out(x, t, W, samples, reduce='mean'):
    if False:
        return 10
    "BlackOut loss function.\n\n    BlackOut loss function is defined as\n\n    .. math::\n\n      -\\log(p(t)) - \\sum_{s \\in S} \\log(1 - p(s)),\n\n    where :math:`t` is the correct label, :math:`S` is a set of negative\n    examples and :math:`p(\\cdot)` is likelihood of a given label.\n    And, :math:`p` is defined as\n\n    .. math::\n\n       p(y) = \\frac{\\exp(W_y^\\top x)}{\n       \\sum_{s \\in samples} \\exp(W_s^\\top x)}.\n\n    The output is a variable whose value depends on the value of\n    the option ``reduce``. If it is ``'no'``, it holds the\n    no loss values. If it is ``'mean'``, this function takes\n    a mean of loss values.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Batch of input vectors.\n            Its shape should be :math:`(N, D)`.\n        t (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Vector of ground truth labels.\n            Its shape should be :math:`(N,)`. Each elements :math:`v`\n            should satisfy :math:`0 \\geq v \\geq V` or :math:`-1`\n            where :math:`V` is the number of label types.\n        W (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Weight matrix.\n            Its shape should be :math:`(V, D)`\n        samples (~chainer.Variable): Negative samples.\n            Its shape should be :math:`(N, S)` where :math:`S` is\n            the number of negative samples.\n        reduce (str): Reduction option. Its value must be either\n            ``'no'`` or ``'mean'``. Otherwise,\n            :class:`ValueError` is raised.\n\n    Returns:\n        ~chainer.Variable:\n            A variable object holding loss value(s).\n            If ``reduce`` is ``'no'``, the output variable holds an\n            array whose shape is :math:`(N,)` .\n            If it is ``'mean'``, it holds a scalar.\n\n    See: `BlackOut: Speeding up Recurrent Neural Network Language Models With\n    Very Large Vocabularies <https://arxiv.org/abs/1511.06909>`_\n\n    .. seealso::\n\n        :class:`~chainer.links.BlackOut` to manage the model parameter ``W``.\n\n    "
    batch_size = x.shape[0]
    neg_emb = embed_id.embed_id(samples, W)
    neg_y = matmul.matmul(neg_emb, x[:, :, None])
    neg_y = reshape.reshape(neg_y, neg_y.shape[:-1])
    pos_emb = expand_dims.expand_dims(embed_id.embed_id(t, W), 1)
    pos_y = matmul.matmul(pos_emb, x[:, :, None])
    pos_y = reshape.reshape(pos_y, pos_y.shape[:-1])
    logz = logsumexp.logsumexp(concat.concat([pos_y, neg_y]), axis=1)
    (blogz, bneg_y) = broadcast.broadcast(reshape.reshape(logz, (batch_size, 1)), neg_y)
    ny = exponential.log(1 - exponential.exp(bneg_y - blogz))
    py = reshape.reshape(pos_y, (batch_size,))
    loss = -(py - logz + _sum.sum(ny, axis=1))
    if reduce == 'mean':
        loss = average.average(loss)
    return loss