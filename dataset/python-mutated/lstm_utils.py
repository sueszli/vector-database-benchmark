import copy
import operator
import torch
from typing import Any, Callable, Optional, Tuple
from torch.ao.quantization import default_weight_observer, default_weight_fake_quant, FakeQuantizeBase, QConfig, QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import convert_to_reference_fx, prepare_fx

def _get_lstm_with_individually_observed_parts(float_lstm: torch.nn.LSTM, example_inputs: Tuple[Any, ...], backend_config: Optional[BackendConfig]=None, linear_output_obs_ctr: Optional[_PartialWrapper]=None, sigmoid_obs_ctr: Optional[_PartialWrapper]=None, tanh_obs_ctr: Optional[_PartialWrapper]=None, cell_state_obs_ctr: Optional[_PartialWrapper]=None, hidden_state_obs_ctr: Optional[_PartialWrapper]=None) -> torch.ao.nn.quantizable.LSTM:
    if False:
        return 10
    "\n    Return an observed `torch.ao.nn.quantizable.LSTM` created from a `torch.nn.LSTM`\n    with specific observers or fake quantizes assigned to the inner ops or submodules.\n\n    In both eager and FX graph mode quantization, `torch.ao.nn.quantizable.LSTM` is\n    used as an observed custom module, which is responsible for inserting its own\n    observers. By default, all inner ops inherit the parent custom module's QConfig.\n    Users who wish to override this behavior may extend `torch.ao.nn.quantizable.LSTM`\n    and use this helper function to customize the observer insertion logic.\n\n    This is meant to be used to convert a float module to an observed module in the\n    custom module flow.\n\n    Args:\n        `float_lstm`: The float LSTM module\n        `example_inputs`: example inputs for the forward function of the LSTM module\n        `backend_config`: BackendConfig to use to observe the LSTM module\n        `linear_output_obs_ctr`: observer or fake quantize for linear outputs Wx + b,\n            where W is the weight matrix, b is the bias, and x is either the inputs\n            or the hidden state from the previous layer (if any)\n        `sigmoid_obs_ctr`: observer or fake quantize for sigmoid activations\n        `tanh_obs_ctr`: observer or fake quantize for tanh activations\n        `cell_state_obs_ctr`: observer or fake quantize for the cell state\n        `hidden_state_obs_ctr`: observer or fake quantize for the hidden state and\n            the output\n\n    Return:\n        A `torch.ao.nn.quantizable.LSTM` with the specified observers or fake quantizes\n        assigned to the inner ops.\n    "

    def make_qconfig(obs_ctr: _PartialWrapper) -> QConfig:
        if False:
            i = 10
            return i + 15
        '\n        Make a QConfig with fixed qparams observers or fake quantizes.\n        '
        if isinstance(obs_ctr(), FakeQuantizeBase):
            weight = default_weight_fake_quant
        else:
            weight = default_weight_observer
        return QConfig(activation=obs_ctr, weight=weight)
    quantizable_lstm = torch.ao.nn.quantizable.LSTM(float_lstm.input_size, float_lstm.hidden_size, float_lstm.num_layers, float_lstm.bias, float_lstm.batch_first, float_lstm.dropout, float_lstm.bidirectional)
    quantizable_lstm.qconfig = float_lstm.qconfig
    for idx in range(float_lstm.num_layers):
        quantizable_lstm.layers[idx] = torch.ao.nn.quantizable.modules.rnn._LSTMLayer.from_float(float_lstm, idx, float_lstm.qconfig, batch_first=False)
    cell_qm = QConfigMapping().set_global(float_lstm.qconfig)
    if sigmoid_obs_ctr is not None:
        cell_qm.set_module_name('input_gate', make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name('forget_gate', make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name('output_gate', make_qconfig(sigmoid_obs_ctr))
    if tanh_obs_ctr is not None:
        cell_qm.set_module_name('cell_gate', make_qconfig(tanh_obs_ctr))
    for layer in quantizable_lstm.layers:
        cell = layer.layer_fw.cell
        cell = prepare_fx(cell, cell_qm, example_inputs, backend_config=backend_config)
        op_index_to_activation_post_process_ctr = {(torch.add, 0): linear_output_obs_ctr, (torch.mul, 0): cell_state_obs_ctr, (torch.mul, 1): cell_state_obs_ctr, (torch.add, 1): cell_state_obs_ctr, (torch.mul, 2): hidden_state_obs_ctr}
        add_count = 0
        mul_count = 0
        for node in cell.graph.nodes:
            op_index: Optional[Tuple[Callable, int]] = None
            if node.target == torch.add:
                op_index = (torch.add, add_count)
                add_count += 1
            elif node.target == torch.mul:
                op_index = (torch.mul, mul_count)
                mul_count += 1
            else:
                continue
            if op_index not in op_index_to_activation_post_process_ctr:
                continue
            assert len(node.users) == 1
            activation_post_process_name = next(iter(node.users.keys())).name
            activation_post_process_ctr = op_index_to_activation_post_process_ctr[op_index]
            if activation_post_process_ctr is not None:
                setattr(cell, activation_post_process_name, activation_post_process_ctr())
        layer.layer_fw.cell = cell
    return quantizable_lstm

def _get_reference_quantized_lstm_module(observed_lstm: torch.ao.nn.quantizable.LSTM, backend_config: Optional[BackendConfig]=None) -> torch.ao.nn.quantized.LSTM:
    if False:
        return 10
    '\n    Return a `torch.ao.nn.quantized.LSTM` created from a `torch.ao.nn.quantizable.LSTM`\n    with observers or fake quantizes inserted through `prepare_fx`, e.g. from\n    `_get_lstm_with_individually_observed_parts`.\n\n    This is meant to be used to convert an observed module to a quantized module in the\n    custom module flow.\n\n    Args:\n        `observed_lstm`: a `torch.ao.nn.quantizable.LSTM` observed through `prepare_fx`\n        `backend_config`: BackendConfig to use to produce the reference quantized model\n\n    Return:\n        A reference `torch.ao.nn.quantized.LSTM` module.\n    '
    quantized_lstm = torch.ao.nn.quantized.LSTM(observed_lstm.input_size, observed_lstm.hidden_size, observed_lstm.num_layers, observed_lstm.bias, observed_lstm.batch_first, observed_lstm.dropout, observed_lstm.bidirectional)
    for (i, layer) in enumerate(quantized_lstm.layers):
        cell = copy.deepcopy(observed_lstm.layers.get_submodule(str(i)).layer_fw.cell)
        cell = convert_to_reference_fx(cell, backend_config=backend_config)
        assert isinstance(cell, torch.fx.GraphModule)
        for node in cell.graph.nodes:
            if node.target == torch.quantize_per_tensor:
                arg = node.args[0]
                if arg.target == 'x' or (arg.target == operator.getitem and arg.args[0].target == 'hidden'):
                    with cell.graph.inserting_before(node):
                        node.replace_all_uses_with(arg)
                        cell.graph.erase_node(node)
            if node.target == 'output':
                for arg in node.args[0]:
                    with cell.graph.inserting_before(node):
                        node.replace_input_with(arg, arg.args[0])
        cell.graph.eliminate_dead_code()
        cell.recompile()
        layer.layer_fw.cell = cell
    return quantized_lstm