from inspect import isclass
import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape

class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def forward(self, val):
        if False:
            for i in range(10):
                print('nop')
        return torch.exp(val)

class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        if False:
            i = 10
            return i + 15
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        if False:
            return 10
        if len(input_args) == 1:
            input_args = input_args[0]
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)

class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        if False:
            while True:
                i = 10
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        if False:
            return 10
        return [mm.forward(*args, **kwargs) for mm in self]

def call_nn_op(op):
    if False:
        for i in range(10):
            print('nop')
    '\n    a helper function that adds appropriate parameters when calling\n    an nn module representing an operation like Softmax\n\n    :param op: the nn.Module operation to instantiate\n    :return: instantiation of the op module with appropriate parameters\n    '
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()

class MLP(nn.Module):

    def __init__(self, mlp_sizes, activation=nn.ReLU, output_activation=None, post_layer_fct=lambda layer_ix, total_layers, layer: None, post_act_fct=lambda layer_ix, total_layers, layer: None, allow_broadcast=False, use_cuda=False):
        if False:
            print('Hello World!')
        super().__init__()
        assert len(mlp_sizes) >= 2, 'Must have input and output layer sizes defined'
        (input_size, hidden_sizes, output_size) = (mlp_sizes[0], mlp_sizes[1:-1], mlp_sizes[-1])
        assert isinstance(input_size, (int, list, tuple)), 'input_size must be int, list, tuple'
        last_layer_size = input_size if type(input_size) == int else sum(input_size)
        all_modules = [ConcatModule(allow_broadcast)]
        for (layer_ix, layer_size) in enumerate(hidden_sizes):
            assert type(layer_size) == int, 'Hidden layer sizes must be ints'
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)
            all_modules.append(cur_linear_layer)
            post_linear = post_layer_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])
            if post_linear is not None:
                all_modules.append(post_linear)
            all_modules.append(activation())
            post_activation = post_act_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])
            if post_activation is not None:
                all_modules.append(post_activation)
            last_layer_size = layer_size
        assert isinstance(output_size, (int, list, tuple)), 'output_size must be int, list, tuple'
        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(call_nn_op(output_activation) if isclass(output_activation) else output_activation)
        else:
            out_layers = []
            for (out_ix, out_size) in enumerate(output_size):
                split_layer = []
                split_layer.append(nn.Linear(last_layer_size, out_size))
                act_out_fct = output_activation if not isinstance(output_activation, (list, tuple)) else output_activation[out_ix]
                if act_out_fct:
                    split_layer.append(call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct)
                out_layers.append(nn.Sequential(*split_layer))
            all_modules.append(ListOutModule(out_layers))
        self.sequential_mlp = nn.Sequential(*all_modules)

    def forward(self, *args, **kwargs):
        if False:
            return 10
        return self.sequential_mlp.forward(*args, **kwargs)