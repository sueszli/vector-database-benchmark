"""Utility functions related to sequence decoders."""
from typing import Dict, Tuple
import torch
from ludwig.constants import ENCODER_OUTPUT_STATE, HIDDEN
from ludwig.modules.reduction_modules import SequenceReducer

def repeat_2D_tensor(tensor, k):
    if False:
        print('Hello World!')
    'Repeats a 2D-tensor k times over the first dimension.\n\n    For example:\n    Input: Tensor of [batch_size, state_size], k=2\n    Output: Tensor of [k, batch_size, state_size]\n    '
    if len(tensor.size()) > 2:
        raise ValueError('Cannot repeat a non-2D tensor with this method.')
    return tensor.repeat(k, 1, 1)

def get_rnn_init_state(combiner_outputs: Dict[str, torch.Tensor], sequence_reducer: SequenceReducer, num_layers: int) -> torch.Tensor:
    if False:
        while True:
            i = 10
    'Computes the hidden state that the RNN decoder should start with.\n\n    Args:\n        combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.\n        sequence_reducer: SequenceReducer to reduce rank-3 to rank-2.\n        num_layers: Number of layers the decoder uses.\n\n    Returns:\n        Tensor of [num_layers, batch_size, hidden_size].\n    '
    if ENCODER_OUTPUT_STATE not in combiner_outputs:
        encoder_output_state = combiner_outputs[HIDDEN]
    else:
        encoder_output_state = combiner_outputs[ENCODER_OUTPUT_STATE]
        if isinstance(encoder_output_state, tuple):
            if len(encoder_output_state) == 2:
                encoder_output_state = encoder_output_state[0]
            elif len(encoder_output_state) == 4:
                encoder_output_state = torch.mean([encoder_output_state[0], encoder_output_state[2]])
            else:
                raise ValueError(f'Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder ' + f'state: {encoder_output_state.size()} that was invalid. Please double check the compatibility ' + 'of your encoder and decoder.')
    if len(encoder_output_state.size()) > 3:
        raise ValueError('Init state for RNN decoders only works for 1d or 2d tensors (encoder_output).')
    if len(encoder_output_state.size()) == 3:
        encoder_output_state = sequence_reducer(encoder_output_state)
    return repeat_2D_tensor(encoder_output_state, num_layers)

def get_lstm_init_state(combiner_outputs: Dict[str, torch.Tensor], sequence_reducer: SequenceReducer, num_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the states that the LSTM decoder should start with.\n\n    Args:\n        combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.\n        sequence_reducer: SequenceReducer to reduce rank-3 to rank-2.\n        num_layers: Number of layers the decoder uses.\n\n    Returns:\n        Tuple of 2 tensors (decoder hidden state, decoder cell state), each [num_layers, batch_size, hidden_size].\n    '
    if ENCODER_OUTPUT_STATE not in combiner_outputs:
        decoder_hidden_state = combiner_outputs[HIDDEN]
        decoder_cell_state = torch.clone(decoder_hidden_state)
    else:
        encoder_output_state = combiner_outputs[ENCODER_OUTPUT_STATE]
        if not isinstance(encoder_output_state, tuple):
            decoder_hidden_state = encoder_output_state
            decoder_cell_state = decoder_hidden_state
        elif len(encoder_output_state) == 2:
            (decoder_hidden_state, decoder_cell_state) = encoder_output_state
        elif len(encoder_output_state) == 4:
            decoder_hidden_state = torch.mean([encoder_output_state[0], encoder_output_state[2]])
            decoder_cell_state = torch.mean([encoder_output_state[1], encoder_output_state[3]])
        else:
            raise ValueError(f'Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder ' + f'state: {encoder_output_state} that was invalid. Please double check the compatibility of your ' + 'encoder and decoder.')
    if len(decoder_hidden_state.size()) > 3 or len(decoder_cell_state.size()) > 3:
        raise ValueError(f'Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder ' + f'state: {decoder_hidden_state.size()} that was invalid. Please double check the compatibility ' + 'of your encoder and decoder.')
    if len(decoder_hidden_state.size()) == 3:
        decoder_hidden_state = sequence_reducer(decoder_hidden_state)
    if len(decoder_cell_state.size()) == 3:
        decoder_cell_state = sequence_reducer(decoder_cell_state)
    return (repeat_2D_tensor(decoder_hidden_state, num_layers), repeat_2D_tensor(decoder_cell_state, num_layers))