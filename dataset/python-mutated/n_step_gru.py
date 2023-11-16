import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.connection import linear
from chainer.functions.rnn import n_step_rnn
from chainer.utils import argument
from chainer import variable
import chainerx
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn

def _extract_apply_in_data(inputs):
    if False:
        for i in range(10):
            print('nop')
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

def _combine_inputs(hx, ws, bs, xs, num_layers, directions):
    if False:
        while True:
            i = 10
    combined = []
    combined.append(hx)
    for x in xs:
        combined.append(x)
    for n in range(num_layers):
        for direction in range(directions):
            idx = directions * n + direction
            for i in range(6):
                combined.append(ws[idx][i])
            for i in range(6):
                combined.append(bs[idx][i])
    return combined

def _seperate_inputs(combined, num_layers, seq_length, directions):
    if False:
        return 10
    hx = combined[0]
    xs = combined[1:1 + seq_length]
    ws = []
    bs = []
    index = 1 + seq_length
    for n in range(num_layers):
        ws.append(combined[index:index + 6])
        bs.append(combined[index + 6:index + 12])
        index += 12
        if directions == 2:
            ws.append(combined[index:index + 6])
            bs.append(combined[index + 6:index + 12])
            index += 12
    return (hx, ws, bs, xs)

class NStepGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states, lengths, rnn_dir='uni', rnn_mode='gru', **kwargs)

class NStepBiGRU(n_step_rnn.BaseNStepRNN):

    def __init__(self, n_layers, states, lengths, **kwargs):
        if False:
            print('Hello World!')
        n_step_rnn.BaseNStepRNN.__init__(self, n_layers, states, lengths, rnn_dir='bi', rnn_mode='gru', **kwargs)

def n_step_gru(n_layers, dropout_ratio, hx, ws, bs, xs, **kwargs):
    if False:
        while True:
            i = 10
    "n_step_gru(n_layers, dropout_ratio, hx, ws, bs, xs)\n\n    Stacked Uni-directional Gated Recurrent Unit function.\n\n    This function calculates stacked Uni-directional GRU with sequences.\n    This function gets an initial hidden state :math:`h_0`, an input\n    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.\n    This function calculates hidden states :math:`h_t` for each time :math:`t`\n    from input :math:`x_t`.\n\n    .. math::\n       r_t &= \\sigma(W_0 x_t + W_3 h_{t-1} + b_0 + b_3) \\\\\n       z_t &= \\sigma(W_1 x_t + W_4 h_{t-1} + b_1 + b_4) \\\\\n       h'_t &= \\tanh(W_2 x_t + b_2 + r_t \\cdot (W_5 h_{t-1} + b_5)) \\\\\n       h_t &= (1 - z_t) \\cdot h'_t + z_t \\cdot h_{t-1}\n\n    As the function accepts a sequence, it calculates :math:`h_t` for all\n    :math:`t` with one call. Six weight matrices and six bias vectors are\n    required for each layers. So, when :math:`S` layers exists, you need to\n    prepare :math:`6S` weight matrices and :math:`6S` bias vectors.\n\n    If the number of layers ``n_layers`` is greather than :math:`1`, input\n    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.\n    Note that all input variables except first layer may have different shape\n    from the first layer.\n\n    Args:\n        n_layers(int): Number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (~chainer.Variable):\n            Variable holding stacked hidden states.\n            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is\n            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is\n            dimension of hidden units.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[i]`` represents weights for i-th layer.\n            Each ``ws[i]`` is a list containing six matrices.\n            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.\n            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(N, I)`` shape as they\n            are multiplied with input variables. All other matrices has\n            ``(N, N)`` shape.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[i]`` represnents biases for i-th layer.\n            Each ``bs[i]`` is a list containing six vectors.\n            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.\n            Shape of each matrix is ``(N,)`` where ``N`` is dimension of\n            hidden units.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable`\n            holding input values. Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is\n            mini-batch size for time ``t``, and ``I`` is size of input units.\n            Note that this function supports variable length sequences.\n            When sequneces has different lengths, sort sequences in descending\n            order by length, and transpose the sorted sequence.\n            :func:`~chainer.functions.transpose_sequence` transpose a list\n            of :func:`~chainer.Variable` holding sequence.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n\n    Returns:\n        tuple: This function returns a tuple containing two elements,\n        ``hy`` and ``ys``.\n\n        - ``hy`` is an updated hidden states whose shape is same as ``hx``.\n        - ``ys`` is a list of :class:`~chainer.Variable` . Each element\n          ``ys[t]`` holds hidden states of the last layer corresponding\n          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is\n          mini-batch size for time ``t``, and ``N`` is size of hidden\n          units. Note that ``B_t`` is the same value as ``xs[t]``.\n\n    "
    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, use_bi_direction=False, **kwargs)

def n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs, **kwargs):
    if False:
        while True:
            i = 10
    "n_step_bigru(n_layers, dropout_ratio, hx, ws, bs, xs)\n\n    Stacked Bi-directional Gated Recurrent Unit function.\n\n    This function calculates stacked Bi-directional GRU with sequences.\n    This function gets an initial hidden state :math:`h_0`, an input\n    sequence :math:`x`, weight matrices :math:`W`, and bias vectors :math:`b`.\n    This function calculates hidden states :math:`h_t` for each time :math:`t`\n    from input :math:`x_t`.\n\n    .. math::\n       r^{f}_t &= \\sigma(W^{f}_0 x_t + W^{f}_3 h_{t-1} + b^{f}_0 + b^{f}_3)\n       \\\\\n       z^{f}_t &= \\sigma(W^{f}_1 x_t + W^{f}_4 h_{t-1} + b^{f}_1 + b^{f}_4)\n       \\\\\n       h^{f'}_t &= \\tanh(W^{f}_2 x_t + b^{f}_2 + r^{f}_t \\cdot (W^{f}_5\n       h_{t-1} + b^{f}_5)) \\\\\n       h^{f}_t &= (1 - z^{f}_t) \\cdot h^{f'}_t + z^{f}_t \\cdot h_{t-1}\n       \\\\\n       r^{b}_t &= \\sigma(W^{b}_0 x_t + W^{b}_3 h_{t-1} + b^{b}_0 + b^{b}_3)\n       \\\\\n       z^{b}_t &= \\sigma(W^{b}_1 x_t + W^{b}_4 h_{t-1} + b^{b}_1 + b^{b}_4)\n       \\\\\n       h^{b'}_t &= \\tanh(W^{b}_2 x_t + b^{b}_2 + r^{b}_t \\cdot (W^{b}_5\n       h_{t-1} + b^{b}_5)) \\\\\n       h^{b}_t &= (1 - z^{b}_t) \\cdot h^{b'}_t + z^{b}_t \\cdot h_{t-1}\n       \\\\\n       h_t  &= [h^{f}_t; h^{b}_t] \\\\\n\n    where :math:`W^{f}` is weight matrices for forward-GRU, :math:`W^{b}` is\n    weight matrices for backward-GRU.\n\n    As the function accepts a sequence, it calculates :math:`h_t` for all\n    :math:`t` with one call. Six weight matrices and six bias vectors are\n    required for each layers. So, when :math:`S` layers exists, you need to\n    prepare :math:`6S` weight matrices and :math:`6S` bias vectors.\n\n    If the number of layers ``n_layers`` is greather than :math:`1`, input\n    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.\n    Note that all input variables except first layer may have different shape\n    from the first layer.\n\n    Args:\n        n_layers(int): Number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (:class:`~chainer.Variable`):\n            Variable holding stacked hidden states.\n            Its shape is ``(2S, B, N)`` where ``S`` is number of layers and is\n            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is\n            dimension of hidden units.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[i]`` represents weights for i-th layer.\n            Each ``ws[i]`` is a list containing six matrices.\n            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.\n            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(N, I)`` shape as they\n            are multiplied with input variables. All other matrices has\n            ``(N, N)`` shape.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[i]`` represnents biases for i-th layer.\n            Each ``bs[i]`` is a list containing six vectors.\n            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.\n            Shape of each matrix is ``(N,)`` where ``N`` is dimension of\n            hidden units.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable` holding input values.\n            Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is\n            mini-batch size for time ``t``, and ``I`` is size of input units.\n            Note that this function supports variable length sequences.\n            When sequneces has different lengths, sort sequences in descending\n            order by length, and transpose the sorted sequence.\n            :func:`~chainer.functions.transpose_sequence` transpose a list\n            of :func:`~chainer.Variable` holding sequence.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n        use_bi_direction (bool): If ``True``, this function uses\n            Bi-direction GRU.\n\n    Returns:\n        tuple: This function returns a tuple containing three elements,\n        ``hy`` and ``ys``.\n\n        - ``hy`` is an updated hidden states whose shape is same as ``hx``.\n        - ``ys`` is a list of :class:`~chainer.Variable` . Each element\n          ``ys[t]`` holds hidden states of the last layer corresponding\n          to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is\n          mini-batch size for time ``t``, and ``N`` is size of hidden\n          units. Note that ``B_t`` is the same value as ``xs[t]``.\n\n    "
    return n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, use_bi_direction=True, **kwargs)

def n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, use_bi_direction, **kwargs):
    if False:
        i = 10
        return i + 15
    "n_step_gru_base(n_layers, dropout_ratio, hx, ws, bs, xs, use_bi_direction)\n\n    Base function for Stack GRU/BiGRU functions.\n\n    This function is used at  :func:`chainer.functions.n_step_bigru` and\n    :func:`chainer.functions.n_step_gru`.\n    This function's behavior depends on argument ``use_bi_direction``.\n\n    Args:\n        n_layers(int): Number of layers.\n        dropout_ratio(float): Dropout ratio.\n        hx (:class:`~chainer.Variable`):\n            Variable holding stacked hidden states.\n            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is\n            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is\n            dimension of hidden units. Because of bi-direction, the\n            first dimension length is ``2S``.\n        ws (list of list of :class:`~chainer.Variable`): Weight matrices.\n            ``ws[i]`` represents weights for i-th layer.\n            Each ``ws[i]`` is a list containing six matrices.\n            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.\n            Only ``ws[0][j]`` where ``0 <= j < 3`` is ``(N, I)`` shape as they\n            are multiplied with input variables. All other matrices has\n            ``(N, N)`` shape.\n        bs (list of list of :class:`~chainer.Variable`): Bias vectors.\n            ``bs[i]`` represnents biases for i-th layer.\n            Each ``bs[i]`` is a list containing six vectors.\n            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.\n            Shape of each matrix is ``(N,)`` where ``N`` is dimension of\n            hidden units.\n        xs (list of :class:`~chainer.Variable`):\n            A list of :class:`~chainer.Variable` holding input values.\n            Each element ``xs[t]`` holds input value\n            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is\n            mini-batch size for time ``t``, and ``I`` is size of input units.\n            Note that this function supports variable length sequences.\n            When sequneces has different lengths, sort sequences in descending\n            order by length, and transpose the sorted sequence.\n            :func:`~chainer.functions.transpose_sequence` transpose a list\n            of :func:`~chainer.Variable` holding sequence.\n            So ``xs`` needs to satisfy\n            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.\n        activation (str): Activation function name.\n            Please select ``tanh`` or ``relu``.\n        use_bi_direction (bool): If ``True``, this function uses\n            Bi-direction GRU.\n\n    .. seealso::\n       :func:`chainer.functions.n_step_rnn`\n       :func:`chainer.functions.n_step_birnn`\n\n    "
    if kwargs:
        argument.check_unexpected_kwargs(kwargs, train='train argument is not supported anymore. Use chainer.using_config', use_cudnn='use_cudnn argument is not supported anymore. Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)
    xp = backend.get_array_module(hx, hx.data)
    directions = 1
    if use_bi_direction:
        directions = 2
    combined = _combine_inputs(hx, ws, bs, xs, n_layers, directions)
    (has_chainerx_array, combined) = _extract_apply_in_data(combined)
    (hx_chx, ws_chx, bs_chx, xs_chx) = _seperate_inputs(combined, n_layers, len(xs), directions)
    if has_chainerx_array and xp is chainerx and (dropout_ratio == 0):
        if use_bi_direction:
            (hy, ys) = chainerx.n_step_bigru(n_layers, hx_chx, ws_chx, bs_chx, xs_chx)
        else:
            (hy, ys) = chainerx.n_step_gru(n_layers, hx_chx, ws_chx, bs_chx, xs_chx)
        hy = variable.Variable._init_unchecked(hy, requires_grad=hy.is_backprop_required(), is_chainerx_array=True)
        ys = [variable.Variable._init_unchecked(y, requires_grad=y.is_backprop_required(), is_chainerx_array=True) for y in ys]
        return (hy, ys)
    if xp is cuda.cupy and chainer.should_use_cudnn('>=auto', 5000):
        lengths = [len(x) for x in xs]
        xs = chainer.functions.concat(xs, axis=0)
        with chainer.using_device(xs.device):
            states = cuda.get_cudnn_dropout_states()
            states.set_dropout_ratio(dropout_ratio)
        w = n_step_rnn.cudnn_rnn_weight_concat(n_layers, states, use_bi_direction, 'gru', ws, bs)
        if use_bi_direction:
            rnn = NStepBiGRU
        else:
            rnn = NStepGRU
        (hy, ys) = rnn(n_layers, states, lengths)(hx, w, xs)
        sections = numpy.cumsum(lengths[:-1])
        ys = chainer.functions.split_axis(ys, sections, 0)
        return (hy, ys)
    else:
        (hy, _, ys) = n_step_rnn.n_step_rnn_impl(_gru, n_layers, dropout_ratio, hx, None, ws, bs, xs, use_bi_direction)
        return (hy, ys)

def _gru(x, h, c, w, b):
    if False:
        print('Hello World!')
    xw = concat.concat([w[0], w[1], w[2]], axis=0)
    hw = concat.concat([w[3], w[4], w[5]], axis=0)
    xb = concat.concat([b[0], b[1], b[2]], axis=0)
    hb = concat.concat([b[3], b[4], b[5]], axis=0)
    gru_x = linear.linear(x, xw, xb)
    gru_h = linear.linear(h, hw, hb)
    (W_r_x, W_z_x, W_x) = split_axis.split_axis(gru_x, 3, axis=1)
    (U_r_h, U_z_h, U_x) = split_axis.split_axis(gru_h, 3, axis=1)
    r = sigmoid.sigmoid(W_r_x + U_r_h)
    z = sigmoid.sigmoid(W_z_x + U_z_h)
    h_bar = tanh.tanh(W_x + r * U_x)
    return ((1 - z) * h_bar + z * h, None)