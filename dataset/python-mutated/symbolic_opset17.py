"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""
import functools
from typing import Optional, Sequence
import torch
from torch import _C
from torch.onnx import _type_utils, errors, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration
__all__ = ['layer_norm', 'stft']
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)

@_onnx_symbolic('aten::layer_norm')
@symbolic_helper.parse_args('v', 'is', 'v', 'v', 'f', 'none')
def layer_norm(g: jit_utils.GraphContext, input: _C.Value, normalized_shape: Sequence[int], weight: _C.Value, bias: _C.Value, eps: float, cudnn_enable: bool):
    if False:
        i = 10
        return i + 15
    axis = -len(normalized_shape)
    return g.op('LayerNormalization', input, weight, bias, epsilon_f=eps, axis_i=axis)

def _compute_edge_sizes(n_fft, window_size):
    if False:
        return 10
    'Helper function to compute the sizes of the edges (left and right)\n    of a given window centered within an FFT size.'
    left = (n_fft - window_size) // 2
    right = n_fft - left - window_size
    return (left, right)

@_onnx_symbolic('aten::stft')
@symbolic_helper.parse_args('v', 'i', 'i', 'i', 'v', 'b', 'b', 'b')
@_beartype.beartype
def stft(g: jit_utils.GraphContext, input: _C.Value, n_fft: int, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: Optional[_C.Value]=None, normalized: bool=False, onesided: Optional[bool]=True, return_complex: Optional[bool]=False) -> _C.Value:
    if False:
        while True:
            i = 10
    'Associates `torch.stft` with the `STFT` ONNX operator.\n    Note that torch.stft calls _VF.stft, without centering or padding options.\n    Hence, this function does not contain these two arguments.\n    See torch.stft source code for more info.\n\n    Args:\n        g: Graph to write the ONNX representation into\n        input: Input tensor for the transformation\n        n_fft: FFT size\n        hop_length: Size of the hop. Defaults to `floot(n_fft // 4)`\n        win_length: Size of the analysis window. Defaults to `n_fft`\n        window: Analysis window. Defaults to a window of all ones\n        normalized: Whether to return a normalized STFT\n        onesided: Whether to return only half (+1) of the results, given the\n            symmetry of the STFT\n        return_complex: Whether to return the complex value (Note: Must be\n            `False` or `None`)\n\n    Returns:\n        op: Operator for torch.stft associated with STFT (ONNX)\n    '
    if return_complex:
        raise errors.SymbolicValueError(msg='STFT does not currently support complex types', value=input)
    frame_step_value = hop_length if hop_length is not None else n_fft // 4
    frame_step_const = g.op('Constant', value_t=torch.tensor(frame_step_value, dtype=torch.int64))
    frame_length_const = g.op('Constant', value_t=torch.tensor(n_fft, dtype=torch.int64))
    signal = input
    signal_rank = symbolic_helper._get_tensor_rank(signal)
    if signal_rank == 1:
        signal = g.op('Unsqueeze', signal, g.op('Constant', value_t=torch.tensor([0], dtype=torch.int64)))
    elif signal_rank > 2:
        raise errors.SymbolicValueError(msg=f'STFT can only take inputs of 1 [signal] or 2 [batch, signal] dimensions. Current rank of signal is {signal_rank}, please reduce it.', value=input)
    n_win = symbolic_helper._get_tensor_dim_size(window, dim=0)
    if n_win is not None:
        win_length_default = win_length if win_length else n_fft
        assert n_win == win_length_default, (f'Analysis window size must equal `win_length` or `n_fft`. Please, set `win_length` or `n_fft` to match `window` size ({n_win})',)
        if n_win < n_fft:
            (left, right) = _compute_edge_sizes(n_fft, n_win)
            left_win = g.op('Constant', value_t=torch.zeros(left))
            right_win = g.op('Constant', value_t=torch.zeros(right))
            window = g.op('Concat', left_win, window, right_win, axis_i=0)
    if symbolic_helper._is_none(window):
        if win_length:
            if win_length > n_fft:
                raise errors.SymbolicValueError(msg=f"The analysis window can't be longer than the size of the FFT. Please set `win_length` ({win_length}) to `n_fft` ({n_fft}) or less.", value=input)
            (left, right) = _compute_edge_sizes(n_fft, win_length)
            torch_window = torch.hstack((torch.zeros(left), torch.ones(win_length), torch.zeros(right)))
        else:
            torch_window = torch.ones(n_fft)
        assert torch_window.shape[0] == n_fft
        window = g.op('Constant', value_t=torch_window)
    window = g.op('Cast', window, to_i=_type_utils.JitScalarType.from_value(signal).onnx_type())
    result = g.op('STFT', signal, frame_step_const, window, frame_length_const, onesided_i=1 if onesided is None or onesided else 0)
    result = g.op('Transpose', result, perm_i=[0, 2, 1, 3])
    if signal_rank == 1:
        result = g.op('Squeeze', result, g.op('Constant', value_t=torch.tensor([0], dtype=torch.int64)))
    if normalized:
        sqrt_nfft = torch.sqrt(torch.tensor(n_fft, dtype=signal.type().dtype()))
        result = g.op('Div', result, g.op('Constant', value_t=sqrt_nfft))
    return result