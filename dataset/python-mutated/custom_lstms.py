import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple
import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
'\nSome helper classes for writing custom TorchScript LSTMs.\n\nGoals:\n- Classes are easy to read, use, and extend\n- Performance of custom LSTMs approach fused-kernel-levels of speed.\n\nA few notes about features we could add to clean up the below code:\n- Support enumerate with nn.ModuleList:\n  https://github.com/pytorch/pytorch/issues/14471\n- Support enumerate/zip with lists:\n  https://github.com/pytorch/pytorch/issues/15952\n- Support overriding of class methods:\n  https://github.com/pytorch/pytorch/issues/10733\n- Support passing around user-defined namedtuple types for readability\n- Support slicing w/ range. It enables reversing lists easily.\n  https://github.com/pytorch/pytorch/issues/10774\n- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose\n  https://github.com/pytorch/pytorch/pull/14922\n'

def script_lstm(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=False, bidirectional=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a ScriptModule that mimics a PyTorch native LSTM.'
    assert bias
    assert not batch_first
    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1
    return stack_type(num_layers, layer_type, first_layer_args=[LSTMCell, input_size, hidden_size], other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size])

def script_lnlstm(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=False, bidirectional=False, decompose_layernorm=False):
    if False:
        while True:
            i = 10
    'Returns a ScriptModule that mimics a PyTorch native LSTM.'
    assert bias
    assert not batch_first
    assert not dropout
    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1
    return stack_type(num_layers, layer_type, first_layer_args=[LayerNormLSTMCell, input_size, hidden_size, decompose_layernorm], other_layer_args=[LayerNormLSTMCell, hidden_size * dirs, hidden_size, decompose_layernorm])
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

def reverse(lst: List[Tensor]) -> List[Tensor]:
    if False:
        for i in range(10):
            print('nop')
    return lst[::-1]

class LSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if False:
            while True:
                i = 10
        (hx, cx) = state
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        (ingate, forgetgate, cellgate, outgate) = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return (hy, (hy, cy))

class LayerNorm(jit.ScriptModule):

    def __init__(self, normalized_shape):
        if False:
            print('Hello World!')
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, input):
        if False:
            while True:
                i = 10
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False)
        return (mu, sigma)

    @jit.script_method
    def forward(self, input):
        if False:
            return 10
        (mu, sigma) = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.weight + self.bias

class LayerNormLSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, decompose_layernorm=False):
        if False:
            return 10
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm
        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if False:
            while True:
                i = 10
        (hx, cx) = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        (ingate, forgetgate, cellgate, outgate) = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = self.layernorm_c(forgetgate * cx + ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return (hy, (hy, cy))

class LSTMLayer(jit.ScriptModule):

    def __init__(self, cell, *cell_args):
        if False:
            while True:
                i = 10
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            (out, state) = self.cell(inputs[i], state)
            outputs += [out]
        return (torch.stack(outputs), state)

class ReverseLSTMLayer(jit.ScriptModule):

    def __init__(self, cell, *cell_args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if False:
            print('Hello World!')
        inputs = reverse(input.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            (out, state) = self.cell(inputs[i], state)
            outputs += [out]
        return (torch.stack(reverse(outputs)), state)

class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.directions = nn.ModuleList([LSTMLayer(cell, *cell_args), ReverseLSTMLayer(cell, *cell_args)])

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        if False:
            return 10
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        i = 0
        for direction in self.directions:
            state = states[i]
            (out, out_state) = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return (torch.cat(outputs, -1), output_states)

def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    if False:
        i = 10
        return i + 15
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        if False:
            print('Hello World!')
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            (output, out_state) = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return (output, output_states)

class StackedLSTM2(jit.ScriptModule):
    __constants__ = ['layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        if False:
            while True:
                i = 10
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)

    @jit.script_method
    def forward(self, input: Tensor, states: List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]:
        if False:
            i = 10
            return i + 15
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            (output, out_state) = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return (output, output_states)

class StackedLSTMWithDropout(jit.ScriptModule):
    __constants__ = ['layers', 'num_layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        if False:
            print('Hello World!')
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)
        self.num_layers = num_layers
        if num_layers == 1:
            warnings.warn('dropout lstm adds dropout layers after all but last recurrent layer, it expects num_layers greater than 1, but got num_layers = 1')
        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        if False:
            print('Hello World!')
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            (output, out_state) = rnn_layer(output, state)
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return (output, output_states)

def flatten_states(states):
    if False:
        for i in range(10):
            print('nop')
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]

def double_flatten_states(states):
    if False:
        print('Hello World!')
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]

def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    if False:
        for i in range(10):
            print('nop')
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    (out, out_state) = rnn(inp, state)
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for (lstm_param, custom_param) in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    (lstm_out, lstm_out_state) = lstm(inp, lstm_state)
    assert (out - lstm_out).abs().max() < 1e-05
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-05
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-05

def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    if False:
        i = 10
        return i + 15
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size)) for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers)
    (out, out_state) = rnn(inp, states)
    custom_state = flatten_states(out_state)
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    lstm_state = flatten_states(states)
    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer:4 * (layer + 1)]
        for (lstm_param, custom_param) in zip(lstm.all_weights[layer], custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    (lstm_out, lstm_out_state) = lstm(inp, lstm_state)
    assert (out - lstm_out).abs().max() < 1e-05
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-05
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-05

def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    if False:
        return 10
    inp = torch.randn(seq_len, batch, input_size)
    states = [[LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size)) for _ in range(2)] for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
    (out, out_state) = rnn(inp, states)
    custom_state = double_flatten_states(out_state)
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index:4 * index + 4]
            for (lstm_param, custom_param) in zip(lstm.all_weights[index], custom_params):
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    (lstm_out, lstm_out_state) = lstm(inp, lstm_state)
    assert (out - lstm_out).abs().max() < 1e-05
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-05
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-05

def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size, num_layers):
    if False:
        return 10
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size)) for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)
    (out, out_state) = rnn(inp, states)

def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size, num_layers):
    if False:
        for i in range(10):
            print('nop')
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size)) for _ in range(num_layers)]
    rnn = script_lnlstm(input_size, hidden_size, num_layers)
    (out, out_state) = rnn(inp, states)
test_script_rnn_layer(5, 2, 3, 7)
test_script_stacked_rnn(5, 2, 3, 7, 4)
test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
test_script_stacked_lnlstm(5, 2, 3, 7, 4)