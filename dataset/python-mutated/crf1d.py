from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import select_item
from chainer.functions.array import split_axis
from chainer.functions.connection import embed_id
from chainer.functions.math import logsumexp
from chainer.functions.math import minmax
from chainer.functions.math import sum as _sum

def crf1d(cost, xs, ys, reduce='mean'):
    if False:
        i = 10
        return i + 15
    "Calculates negative log-likelihood of linear-chain CRF.\n\n    It takes a transition cost matrix, a sequence of costs, and a sequence of\n    labels. Let :math:`c_{st}` be a transition cost from a label :math:`s` to\n    a label :math:`t`, :math:`x_{it}` be a cost of a label :math:`t` at\n    position :math:`i`, and :math:`y_i` be an expected label at position\n    :math:`i`. The negative log-likelihood of linear-chain CRF is defined as\n\n    .. math::\n        L = -\\left( \\sum_{i=1}^l x_{iy_i} + \\\n             \\sum_{i=1}^{l-1} c_{y_i y_{i+1}} - {\\log(Z)} \\right) ,\n\n    where :math:`l` is the length of the input sequence and :math:`Z` is the\n    normalizing constant called partition function.\n\n    .. note::\n\n       When you want to calculate the negative log-likelihood of sequences\n       which have different lengths, sort the sequences in descending order of\n       lengths and transpose the sequences.\n       For example, you have three input sequences:\n\n       >>> a1 = a2 = a3 = a4 = np.random.uniform(-1, 1, 3).astype(np.float32)\n       >>> b1 = b2 = b3 = np.random.uniform(-1, 1, 3).astype(np.float32)\n       >>> c1 = c2 = np.random.uniform(-1, 1, 3).astype(np.float32)\n\n       >>> a = [a1, a2, a3, a4]\n       >>> b = [b1, b2, b3]\n       >>> c = [c1, c2]\n\n       where ``a1`` and all other variables are arrays with ``(K,)`` shape.\n       Make a transpose of the sequences:\n\n       >>> x1 = np.stack([a1, b1, c1])\n       >>> x2 = np.stack([a2, b2, c2])\n       >>> x3 = np.stack([a3, b3])\n       >>> x4 = np.stack([a4])\n\n       and make a list of the arrays:\n\n       >>> xs = [x1, x2, x3, x4]\n\n       You need to make label sequences in the same fashion.\n       And then, call the function:\n\n       >>> cost = chainer.Variable(\n       ...     np.random.uniform(-1, 1, (3, 3)).astype(np.float32))\n       >>> ys = [np.zeros(x.shape[0:1], dtype=np.int32) for x in xs]\n       >>> loss = F.crf1d(cost, xs, ys)\n\n       It calculates mean of the negative log-likelihood of the three\n       sequences.\n\n       The output is a variable whose value depends on the value of\n       the option ``reduce``. If it is ``'no'``, it holds the elementwise\n       loss values. If it is ``'mean'``, it holds mean of the loss values.\n\n\n    Args:\n        cost (:class:`~chainer.Variable` or :ref:`ndarray`):\n            A :math:`K \\times K` matrix which holds transition\n            cost between two labels, where :math:`K` is the number of labels.\n        xs (list of Variable): Input vector for each label.\n            ``len(xs)`` denotes the length of the sequence,\n            and each :class:`~chainer.Variable` holds a :math:`B \\times K`\n            matrix, where :math:`B` is mini-batch size, :math:`K` is the number\n            of labels.\n            Note that :math:`B`\\ s in all the variables are not necessary\n            the same, i.e., it accepts the input sequences with different\n            lengths.\n        ys (list of Variable): Expected output labels. It needs to have the\n            same length as ``xs``. Each :class:`~chainer.Variable` holds a\n            :math:`B` integer vector.\n            When ``x`` in ``xs`` has the different :math:`B`, correspoding\n            ``y`` has the same :math:`B`. In other words, ``ys`` must satisfy\n            ``ys[i].shape == xs[i].shape[0:1]`` for all ``i``.\n        reduce (str): Reduction option. Its value must be either\n            ``'mean'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.\n\n    Returns:\n        ~chainer.Variable: A variable holding the average negative\n        log-likelihood of the input sequences.\n\n    .. note::\n\n        See detail in the original paper: `Conditional Random Fields:\n        Probabilistic Models for Segmenting and Labeling Sequence Data\n        <https://repository.upenn.edu/cis_papers/159/>`_.\n\n    "
    if reduce not in ('mean', 'no'):
        raise ValueError("only 'mean' and 'no' are valid for 'reduce', but '%s' is given" % reduce)
    assert xs[0].shape[1] == cost.shape[0]
    n_label = cost.shape[0]
    n_batch = xs[0].shape[0]
    alpha = xs[0]
    alphas = []
    for x in xs[1:]:
        batch = x.shape[0]
        if alpha.shape[0] > batch:
            (alpha, alpha_rest) = split_axis.split_axis(alpha, [batch], axis=0)
            alphas.append(alpha_rest)
        (b_alpha, b_cost) = broadcast.broadcast(alpha[..., None], cost)
        alpha = logsumexp.logsumexp(b_alpha + b_cost, axis=1) + x
    if alphas:
        alphas.append(alpha)
        alpha = concat.concat(alphas[::-1], axis=0)
    logz = logsumexp.logsumexp(alpha, axis=1)
    cost = reshape.reshape(cost, (cost.size, 1))
    score = select_item.select_item(xs[0], ys[0])
    scores = []
    for (x, y, y_prev) in zip(xs[1:], ys[1:], ys[:-1]):
        batch = x.shape[0]
        if score.shape[0] > batch:
            (y_prev, _) = split_axis.split_axis(y_prev, [batch], axis=0)
            (score, score_rest) = split_axis.split_axis(score, [batch], axis=0)
            scores.append(score_rest)
        score += select_item.select_item(x, y) + reshape.reshape(embed_id.embed_id(y_prev * n_label + y, cost), (batch,))
    if scores:
        scores.append(score)
        score = concat.concat(scores[::-1], axis=0)
    loss = logz - score
    if reduce == 'mean':
        return _sum.sum(loss) / n_batch
    else:
        return loss

def argmax_crf1d(cost, xs):
    if False:
        i = 10
        return i + 15
    'Computes a state that maximizes a joint probability of the given CRF.\n\n    Args:\n        cost (:class:`~chainer.Variable` or :ref:`ndarray`):\n            A :math:`K \\times K` matrix which holds transition\n            cost between two labels, where :math:`K` is the number of labels.\n        xs (list of Variable): Input vector for each label.\n            ``len(xs)`` denotes the length of the sequence,\n            and each :class:`~chainer.Variable` holds a :math:`B \\times K`\n            matrix, where :math:`B` is mini-batch size, :math:`K` is the number\n            of labels.\n            Note that :math:`B`\\ s in all the variables are not necessary\n            the same, i.e., it accepts the input sequences with different\n            lengths.\n\n    Returns:\n        tuple: A tuple of :class:`~chainer.Variable` object ``s`` and a\n        :class:`list` ``ps``.\n        The shape of ``s`` is ``(B,)``, where ``B`` is the mini-batch size.\n        i-th element of ``s``, ``s[i]``, represents log-likelihood of i-th\n        data.\n        ``ps`` is a list of :ref:`ndarray`, and denotes the state that\n        maximizes the point probability.\n        ``len(ps)`` is equal to ``len(xs)``, and shape of each ``ps[i]`` is\n        the mini-batch size of the corresponding ``xs[i]``. That means,\n        ``ps[i].shape == xs[i].shape[0:1]``.\n    '
    alpha = xs[0]
    alphas = []
    max_inds = []
    for x in xs[1:]:
        batch = x.shape[0]
        if alpha.shape[0] > batch:
            (alpha, alpha_rest) = split_axis.split_axis(alpha, [batch], axis=0)
            alphas.append(alpha_rest)
        else:
            alphas.append(None)
        (b_alpha, b_cost) = broadcast.broadcast(alpha[..., None], cost)
        scores = b_alpha + b_cost
        max_ind = minmax.argmax(scores, axis=1)
        max_inds.append(max_ind)
        alpha = minmax.max(scores, axis=1) + x
    inds = minmax.argmax(alpha, axis=1)
    path = [inds.data]
    for (m, a) in zip(max_inds[::-1], alphas[::-1]):
        inds = select_item.select_item(m, inds)
        if a is not None:
            inds = concat.concat([inds, minmax.argmax(a, axis=1)], axis=0)
        path.append(inds.data)
    path.reverse()
    score = minmax.max(alpha, axis=1)
    for a in alphas[::-1]:
        if a is None:
            continue
        score = concat.concat([score, minmax.max(a, axis=1)], axis=0)
    return (score, path)