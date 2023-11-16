import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.functions.array import reshape
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.rnn import lstm
from chainer.functions.rnn import n_step_rnn
from chainer.utils import argument
from chainer import variable
import chainerx
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn

def _extract_apply_in_data(inputs):
    if False:
        while True:
            i = 10
    if not inputs:
        return (False, ())
    if chainerx.is_available():
        has_chainerx_array = False
        arrays = []
        for x in inputs:
            if isinstance(x, variable.Variable):
                if x._has_chainerx_array:
                    arrays.append(x._data[0])
                    has_chainerx_array = True
                else:
                    arrays.append(x.array)
            else:
                arrays.append(x)
                if not has_chainerx_array:
                    if isinstance(x, chainerx.ndarray):
                        has_chainerx_array = True
        return (has_chainerx_array, tuple(arrays))
    else:
        return (False, tuple([x.array if isinstance(x, variable.Variable) else x for x in inputs]))

def _combine_inputs(hx, cx, ws, bs, xs, num_layers, directions):
    if False:
        print('Hello World!')
    combined = []
    combined.append(hx)
    combined.append(cx)
    for x in xs:
        combined.append(x)
    for n in range(num_layers):
        for direction in range(directions):
            idx = directions * n + direction
            for i in range(8):
                combined.append(ws[idx][i])
            for i in range(8):
                combined.append(bs[idx][i])
    return combined

def _seperate_inputs(combined, num_layers, seq_length, directions):
    if False:
        i = 10
        return i + 15
    hx = combined[0]
    cx = combined[1]
    xs = combined[2:2 + seq_length]
    ws = []
    bs = []
    index = 2 + seq_length
    for n in range(num_layers):
        ws.append(combined[index:index + 8])
        bs.append(combined[index + 8:index + 16])
        index += 16
        if directions == 2:
            ws.append(combined[index:index + 8])
            bs.append(combined[index + 8:index + 16])
            index += 16
    return (hx, cx, ws, bs, xs)

def _stack_weight(ws):
    if False:
        print('Hello World!')
    w = stack.stack(ws, axis=1)
    shape = w.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])

class NStepLSTM(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths):
        if False:
            return 10
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states, lengths, rnn_dir='uni', rnn_mode='lstm')

class NStepBiLSTM(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths):
        if False:
            for i in range(10):
                print('nop')
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states, lengths, rnn_dir='bi', rnn_mode='lstm')

def n_step_lstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs, **kwargs):
    if False:
        return 10
    'n_step_lstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs)\n\n    Stacked Uni-directional Long Short-Term Memory function.\n\n    This function calculates stacked Uni-directional LSTM with sequences.\n    This function gets an initial hidden state :math:`h_0`, an initial cell\n    state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,\n    and bias vectors :math:`b`.\n    This function calculates hidden states :math:`h_t` and :math:`c_t` for each\n    time :math:`t` from input :math:`x_t`.\n\n    .. math::\n       i_t &= \\sigma(W_0 x_t + W_4 h_{t-1} + b_0 + b_4) \\\\\n       f_t &= \\sigma(W_1 x_t + W_5 h_{t-1} + b_1 + b_5) \\\\\n       o_t &= \\sigma(W_2 x_t + W_6 h_{t-1} + b_2 + b_6) \\\\\n       a_t &= \\tanh(W_3 x_t + W_7 h_{t-1} + b_3 + b_7) \\\\\n       c_t &= f_t \\cdot c_{t-1} + i_t \\cdot a_t \\\\\n       h_t &= o_t \\cdot \\tanh(c_t)\n\n    As the function accepts a sequence, it calculates :math:`h_t` for all\n    :math:`t` with one call. Eight weight matrices and eight bias vectors are\n    required for each layer. So, when :math:`S` layers exist, you need to\n    prepare :math:`8S` weight matrices and :math:`8S` bias vectors.\n\n    If the number of layers ``n_layers`` is greater than :math:`1`, the input\n    of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.\n    Note that all input variables except the first layer may have different\n    shape from the first layer.\n\n    Args:\n        n_layers(int): The number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (:class:`~chainer.Variable`):\n            Variable holding stacked hidden states.\n            Its shape is ``(S, B, N)`` where ``S`` is the number of layers and\n            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``\n            is the dimension of the hidden units.\n        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.\n            It has the same shape as ``hx``.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[i]`` represents the weights for the i-th layer.\n            Each ``ws[i]`` is a list containing eight matrices.\n            ``ws[i][j]`` corresponds to :math:`W_j` in the equation.\n            Only ``ws[0][j]`` where ``0 <= j < 4`` are ``(N, I)``-shaped as\n            they are multiplied with input variables, where ``I`` is the size\n            of the input and ``N`` is the dimension of the hidden units. All\n            other matrices are ``(N, N)``-shaped.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[i]`` represents the biases for the i-th layer.\n            Each ``bs[i]`` is a list containing eight vectors.\n            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.\n            The shape of each matrix is ``(N,)`` where ``N`` is the dimension\n            of the hidden units.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable`\n            holding input values. Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the\n            mini-batch size for time ``t``. The sequences must be transposed.\n            :func:`~chainer.functions.transpose_sequence` can be used to\n            transpose a list of :class:`~chainer.Variable`\\ s each\n            representing a sequence.\n            When sequences has different lengths, they must be\n            sorted in descending order of their lengths before transposing.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n\n    Returns:\n        tuple: This function returns a tuple containing three elements,\n        ``hy``, ``cy`` and ``ys``.\n\n        - ``hy`` is an updated hidden states whose shape is the same as\n          ``hx``.\n        - ``cy`` is an updated cell states whose shape is the same as\n          ``cx``.\n        - ``ys`` is a list of :class:`~chainer.Variable` . Each element\n          ``ys[t]`` holds hidden states of the last layer corresponding\n          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is\n          the mini-batch size for time ``t``, and ``N`` is size of hidden\n          units. Note that ``B_t`` is the same value as ``xs[t]``.\n\n    .. note::\n\n       The dimension of hidden units is limited to only one size ``N``. If you\n       want to use variable dimension of hidden units, please use\n       :class:`chainer.functions.lstm`.\n\n    .. seealso::\n\n       :func:`chainer.functions.lstm`\n\n    .. admonition:: Example\n\n        >>> batchs = [3, 2, 1]  # support variable length sequences\n        >>> in_size, out_size, n_layers = 3, 2, 2\n        >>> dropout_ratio = 0.0\n        >>> xs = [np.ones((b, in_size)).astype(np.float32) for b in batchs]\n        >>> [x.shape for x in xs]\n        [(3, 3), (2, 3), (1, 3)]\n        >>> h_shape = (n_layers, batchs[0], out_size)\n        >>> hx = np.ones(h_shape).astype(np.float32)\n        >>> cx = np.ones(h_shape).astype(np.float32)\n        >>> w_in = lambda i, j: in_size if i == 0 and j < 4 else out_size\n        >>> ws = []\n        >>> bs = []\n        >>> for n in range(n_layers):\n        ...     ws.append([np.ones((out_size, w_in(n, i))).astype(np.float32) for i in range(8)])\n        ...     bs.append([np.ones((out_size,)).astype(np.float32) for _ in range(8)])\n        ...\n        >>> ws[0][0].shape  # ws[0][:4].shape are (out_size, in_size)\n        (2, 3)\n        >>> ws[1][0].shape  # others are (out_size, out_size)\n        (2, 2)\n        >>> bs[0][0].shape\n        (2,)\n        >>> hy, cy, ys = F.n_step_lstm(\n        ...     n_layers, dropout_ratio, hx, cx, ws, bs, xs)\n        >>> hy.shape\n        (2, 3, 2)\n        >>> cy.shape\n        (2, 3, 2)\n        >>> [y.shape for y in ys]\n        [(3, 2), (2, 2), (1, 2)]\n\n    '
    return n_step_lstm_base(n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction=False, **kwargs)

def n_step_bilstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs, **kwargs):
    if False:
        while True:
            i = 10
    'n_step_bilstm(n_layers, dropout_ratio, hx, cx, ws, bs, xs)\n\n    Stacked Bi-directional Long Short-Term Memory function.\n\n    This function calculates stacked Bi-directional LSTM with sequences.\n    This function gets an initial hidden state :math:`h_0`, an initial cell\n    state :math:`c_0`, an input sequence :math:`x`, weight matrices :math:`W`,\n    and bias vectors :math:`b`.\n    This function calculates hidden states :math:`h_t` and :math:`c_t` for each\n    time :math:`t` from input :math:`x_t`.\n\n    .. math::\n        i^{f}_t &=& \\sigma(W^{f}_0 x_t + W^{f}_4 h_{t-1} + b^{f}_0 + b^{f}_4),\n        \\\\\n        f^{f}_t &=& \\sigma(W^{f}_1 x_t + W^{f}_5 h_{t-1} + b^{f}_1 + b^{f}_5),\n        \\\\\n        o^{f}_t &=& \\sigma(W^{f}_2 x_t + W^{f}_6 h_{t-1} + b^{f}_2 + b^{f}_6),\n        \\\\\n        a^{f}_t &=& \\tanh(W^{f}_3 x_t + W^{f}_7 h_{t-1} + b^{f}_3 + b^{f}_7),\n        \\\\\n        c^{f}_t &=& f^{f}_t \\cdot c^{f}_{t-1} + i^{f}_t \\cdot a^{f}_t,\n        \\\\\n        h^{f}_t &=& o^{f}_t \\cdot \\tanh(c^{f}_t),\n        \\\\\n        i^{b}_t &=& \\sigma(W^{b}_0 x_t + W^{b}_4 h_{t-1} + b^{b}_0 + b^{b}_4),\n        \\\\\n        f^{b}_t &=& \\sigma(W^{b}_1 x_t + W^{b}_5 h_{t-1} + b^{b}_1 + b^{b}_5),\n        \\\\\n        o^{b}_t &=& \\sigma(W^{b}_2 x_t + W^{b}_6 h_{t-1} + b^{b}_2 + b^{b}_6),\n        \\\\\n        a^{b}_t &=& \\tanh(W^{b}_3 x_t + W^{b}_7 h_{t-1} + b^{b}_3 + b^{b}_7),\n        \\\\\n        c^{b}_t &=& f^{b}_t \\cdot c^{b}_{t-1} + i^{b}_t \\cdot a^{b}_t, \\\\\n        h^{b}_t &=& o^{b}_t \\cdot \\tanh(c^{b}_t), \\\\\n        h_t &=& [h^{f}_t; h^{b}_t]\n\n    where :math:`W^{f}` is the weight matrices for forward-LSTM, :math:`W^{b}`\n    is weight matrices for backward-LSTM.\n\n    As the function accepts a sequence, it calculates :math:`h_t` for all\n    :math:`t` with one call. Eight weight matrices and eight bias vectors are\n    required for each layer of each direction. So, when :math:`S` layers\n    exist, you need to prepare :math:`16S` weight matrices and :math:`16S`\n    bias vectors.\n\n    If the number of layers ``n_layers`` is greater than :math:`1`, the input\n    of the ``k``-th layer is the hidden state ``h_t`` of the ``k-1``-th layer.\n    Note that all input variables except the first layer may have different\n    shape from the first layer.\n\n    Args:\n        n_layers(int): The number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (:class:`~chainer.Variable`):\n            Variable holding stacked hidden states.\n            Its shape is ``(2S, B, N)`` where ``S`` is the number of layers and\n            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``\n            is the dimension of the hidden units. Because of bi-direction, the\n            first dimension length is ``2S``.\n        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.\n            It has the same shape as ``hx``.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[2 * l + m]`` represents the weights for the l-th layer of\n            the m-th direction. (``m == 0`` means the forward direction and\n            ``m == 1`` means the backward direction.) Each ``ws[i]`` is a\n            list containing eight matrices. ``ws[i][j]`` corresponds to\n            :math:`W_j` in the equation. ``ws[0][j]`` and ``ws[1][j]`` where\n            ``0 <= j < 4`` are ``(N, I)``-shaped because they are multiplied\n            with input variables, where ``I`` is the size of the input.\n            ``ws[i][j]`` where ``2 <= i`` and ``0 <= j < 4`` are\n            ``(N, 2N)``-shaped because they are multiplied with two hidden\n            layers :math:`h_t = [h^{f}_t; h^{b}_t]`. All other matrices are\n            ``(N, N)``-shaped.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[2 * l + m]`` represents the weights for the l-th layer of\n            m-th direction. (``m == 0`` means the forward direction and\n            ``m == 1`` means the backward direction.)\n            Each ``bs[i]`` is a list containing eight vectors.\n            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.\n            The shape of each matrix is ``(N,)``.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable`\n            holding input values. Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the\n            mini-batch size for time ``t``. The sequences must be transposed.\n            :func:`~chainer.functions.transpose_sequence` can be used to\n            transpose a list of :class:`~chainer.Variable`\\ s each\n            representing a sequence.\n            When sequences has different lengths, they must be\n            sorted in descending order of their lengths before transposing.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n\n    Returns:\n        tuple: This function returns a tuple containing three elements,\n        ``hy``, ``cy`` and ``ys``.\n\n        - ``hy`` is an updated hidden states whose shape is the same as\n          ``hx``.\n        - ``cy`` is an updated cell states whose shape is the same as\n          ``cx``.\n        - ``ys`` is a list of :class:`~chainer.Variable` . Each element\n          ``ys[t]`` holds hidden states of the last layer corresponding\n          to an input ``xs[t]``. Its shape is ``(B_t, 2N)`` where ``B_t``\n          is the mini-batch size for time ``t``, and ``N`` is size of\n          hidden units. Note that ``B_t`` is the same value as ``xs[t]``.\n\n    .. admonition:: Example\n\n        >>> batchs = [3, 2, 1]  # support variable length sequences\n        >>> in_size, out_size, n_layers = 3, 2, 2\n        >>> dropout_ratio = 0.0\n        >>> xs = [np.ones((b, in_size)).astype(np.float32) for b in batchs]\n        >>> [x.shape for x in xs]\n        [(3, 3), (2, 3), (1, 3)]\n        >>> h_shape = (n_layers * 2, batchs[0], out_size)\n        >>> hx = np.ones(h_shape).astype(np.float32)\n        >>> cx = np.ones(h_shape).astype(np.float32)\n        >>> def w_in(i, j):\n        ...     if i == 0 and j < 4:\n        ...         return in_size\n        ...     elif i > 0 and j < 4:\n        ...         return out_size * 2\n        ...     else:\n        ...         return out_size\n        ...\n        >>> ws = []\n        >>> bs = []\n        >>> for n in range(n_layers):\n        ...     for direction in (0, 1):\n        ...         ws.append([np.ones((out_size, w_in(n, i))).astype(np.float32) for i in range(8)])\n        ...         bs.append([np.ones((out_size,)).astype(np.float32) for _ in range(8)])\n        ...\n        >>> ws[0][0].shape  # ws[0:2][:4].shape are (out_size, in_size)\n        (2, 3)\n        >>> ws[2][0].shape  # ws[2:][:4].shape are (out_size, 2 * out_size)\n        (2, 4)\n        >>> ws[0][4].shape  # others are (out_size, out_size)\n        (2, 2)\n        >>> bs[0][0].shape\n        (2,)\n        >>> hy, cy, ys = F.n_step_bilstm(\n        ...     n_layers, dropout_ratio, hx, cx, ws, bs, xs)\n        >>> hy.shape\n        (4, 3, 2)\n        >>> cy.shape\n        (4, 3, 2)\n        >>> [y.shape for y in ys]\n        [(3, 4), (2, 4), (1, 4)]\n\n    '
    return n_step_lstm_base(n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction=True, **kwargs)

def n_step_lstm_base(n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction, **kwargs):
    if False:
        return 10
    "Base function for Stack LSTM/BiLSTM functions.\n\n    This function is used at :func:`chainer.functions.n_step_lstm` and\n    :func:`chainer.functions.n_step_bilstm`.\n    This function's behavior depends on following arguments,\n    ``activation`` and ``use_bi_direction``.\n\n    Args:\n        n_layers(int): The number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (:class:`~chainer.Variable`):\n            Variable holding stacked hidden states.\n            Its shape is ``(S, B, N)`` where ``S`` is the number of layers and\n            is equal to ``n_layers``, ``B`` is the mini-batch size, and ``N``\n            is the dimension of the hidden units.\n        cx (:class:`~chainer.Variable`): Variable holding stacked cell states.\n            It has the same shape as ``hx``.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[i]`` represents the weights for the i-th layer.\n            Each ``ws[i]`` is a list containing eight matrices.\n            ``ws[i][j]`` corresponds to :math:`W_j` in the equation.\n            Only ``ws[0][j]`` where ``0 <= j < 4`` are ``(N, I)``-shape as they\n            are multiplied with input variables, where ``I`` is the size of\n            the input and ``N`` is the dimension of the hidden units. All\n            other matrices are ``(N, N)``-shaped.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[i]`` represents the biases for the i-th layer.\n            Each ``bs[i]`` is a list containing eight vectors.\n            ``bs[i][j]`` corresponds to :math:`b_j` in the equation.\n            The shape of each matrix is ``(N,)``.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable`\n            holding input values. Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is the\n            mini-batch size for time ``t``. The sequences must be transposed.\n            :func:`~chainer.functions.transpose_sequence` can be used to\n            transpose a list of :class:`~chainer.Variable`\\ s each\n            representing a sequence.\n            When sequences has different lengths, they must be\n            sorted in descending order of their lengths before transposing.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n        use_bi_direction (bool): If ``True``, this function uses Bi-directional\n            LSTM.\n\n    Returns:\n        tuple: This function returns a tuple containing three elements,\n        ``hy``, ``cy`` and ``ys``.\n\n            - ``hy`` is an updated hidden states whose shape is the same as\n              ``hx``.\n            - ``cy`` is an updated cell states whose shape is the same as\n              ``cx``.\n            - ``ys`` is a list of :class:`~chainer.Variable` . Each element\n              ``ys[t]`` holds hidden states of the last layer corresponding\n              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is\n              the mini-batch size for time ``t``. Note that ``B_t`` is the same\n              value as ``xs[t]``.\n\n    .. seealso::\n\n       :func:`chainer.functions.n_step_lstm`\n       :func:`chainer.functions.n_step_bilstm`\n\n    "
    if kwargs:
        argument.check_unexpected_kwargs(kwargs, train='train argument is not supported anymore. Use chainer.using_config', use_cudnn='use_cudnn argument is not supported anymore. Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)
    x_in = xs[0].shape[1]
    w_in = ws[0][0].shape[1]
    if x_in != w_in:
        raise ValueError('Inconsistent input size in input values and weight parameters: {} != {}'.format(x_in, w_in))
    xp = backend.get_array_module(hx, hx.data)
    use_cuda = xp is cuda.cupy or (xp is chainerx and hx.device.device.backend.name == 'cuda')
    directions = 1
    if use_bi_direction:
        directions = 2
    combined = _combine_inputs(hx, cx, ws, bs, xs, n_layers, directions)
    (has_chainerx_array, combined) = _extract_apply_in_data(combined)
    (hx_chx, cx_chx, ws_chx, bs_chx, xs_chx) = _seperate_inputs(combined, n_layers, len(xs), directions)
    if has_chainerx_array and xp is chainerx and (dropout_ratio == 0):
        if use_bi_direction:
            (hy, cy, ys) = chainerx.n_step_bilstm(n_layers, hx_chx, cx_chx, ws_chx, bs_chx, xs_chx)
        else:
            (hy, cy, ys) = chainerx.n_step_lstm(n_layers, hx_chx, cx_chx, ws_chx, bs_chx, xs_chx)
        hy = variable.Variable._init_unchecked(hy, requires_grad=hy.is_backprop_required(), is_chainerx_array=True)
        cy = variable.Variable._init_unchecked(cy, requires_grad=cy.is_backprop_required(), is_chainerx_array=True)
        ys = [variable.Variable._init_unchecked(y, requires_grad=y.is_backprop_required(), is_chainerx_array=True) for y in ys]
        return (hy, cy, ys)
    elif use_cuda and chainer.should_use_cudnn('>=auto', 5000):
        lengths = [len(x) for x in xs]
        xs = chainer.functions.concat(xs, axis=0)
        with chainer.using_device(xs.device):
            states = cuda.get_cudnn_dropout_states()
            states.set_dropout_ratio(dropout_ratio)
        w = n_step_rnn.cudnn_rnn_weight_concat(n_layers, states, use_bi_direction, 'lstm', ws, bs)
        if use_bi_direction:
            rnn = NStepBiLSTM
        else:
            rnn = NStepLSTM
        (hy, cy, ys) = rnn(n_layers, states, lengths)(hx, cx, w, xs)
        sections = numpy.cumsum(lengths[:-1])
        ys = chainer.functions.split_axis(ys, sections, 0)
        return (hy, cy, ys)
    else:
        return n_step_rnn.n_step_rnn_impl(_lstm, n_layers, dropout_ratio, hx, cx, ws, bs, xs, use_bi_direction)

def _lstm(x, h, c, w, b):
    if False:
        i = 10
        return i + 15
    xw = _stack_weight([w[2], w[0], w[1], w[3]])
    hw = _stack_weight([w[6], w[4], w[5], w[7]])
    xb = _stack_weight([b[2], b[0], b[1], b[3]])
    hb = _stack_weight([b[6], b[4], b[5], b[7]])
    lstm_in = linear.linear(x, xw, xb) + linear.linear(h, hw, hb)
    (c_bar, h_bar) = lstm.lstm(c, lstm_in)
    return (h_bar, c_bar)