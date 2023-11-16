"""
An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.
"""
from typing import Optional, Tuple, List
import torch
from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal

class LstmCellWithProjection(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.

    [0]: https://arxiv.org/abs/1512.05287

    # Parameters

    input_size : `int`, required.
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required.
        The dimension of the outputs of the LSTM.
    cell_size : `int`, required.
        The dimension of the memory cell used for the LSTM.
    go_forward : `bool`, optional (default = `True`)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        The dropout probability to be used in a dropout scheme as stated in
        [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
        [0]. Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value : `float`, optional, (default = `None`)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value : `float`, optional, (default = `None`)
        The magnitude with which to clip the memory cell.

    # Returns

    output_accumulator : `torch.FloatTensor`
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state : `Tuple[torch.FloatTensor, torch.FloatTensor]`
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(self, input_size: int, hidden_size: int, cell_size: int, go_forward: bool=True, recurrent_dropout_probability: float=0.0, memory_cell_clip_value: Optional[float]=None, state_projection_clip_value: Optional[float]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)
        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if False:
            print('Hello World!')
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self, inputs: torch.FloatTensor, batch_lengths: List[int], initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        if False:
            i = 10
            return i + 15
        '\n        # Parameters\n\n        inputs : `torch.FloatTensor`, required.\n            A tensor of shape (batch_size, num_timesteps, input_size)\n            to apply the LSTM over.\n        batch_lengths : `List[int]`, required.\n            A list of length batch_size containing the lengths of the sequences in batch.\n        initial_state : `Tuple[torch.Tensor, torch.Tensor]`, optional, (default = `None`)\n            A tuple (state, memory) representing the initial hidden state and memory\n            of the LSTM. The `state` has shape (1, batch_size, hidden_size) and the\n            `memory` has shape (1, batch_size, cell_size).\n\n        # Returns\n\n        output_accumulator : `torch.FloatTensor`\n            The outputs of the LSTM for each timestep. A tensor of shape\n            (batch_size, max_timesteps, hidden_size) where for a given batch\n            element, all outputs past the sequence length for that batch are\n            zero tensors.\n        final_state : `Tuple[torch.FloatTensor, torch.FloatTensor]`\n            A tuple (state, memory) representing the initial hidden state and memory\n            of the LSTM. The `state` has shape (1, batch_size, hidden_size) and the\n            `memory` has shape (1, batch_size, cell_size).\n        '
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]
        output_accumulator = inputs.new_zeros(batch_size, total_timesteps, self.hidden_size)
        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_state)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:current_length_index + 1].clone()
            timestep_input = inputs[0:current_length_index + 1, index]
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            input_gate = torch.sigmoid(projected_input[:, 0 * self.cell_size:1 * self.cell_size] + projected_state[:, 0 * self.cell_size:1 * self.cell_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.cell_size:2 * self.cell_size] + projected_state[:, 1 * self.cell_size:2 * self.cell_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.cell_size:3 * self.cell_size] + projected_state[:, 2 * self.cell_size:3 * self.cell_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.cell_size:4 * self.cell_size] + projected_state[:, 3 * self.cell_size:4 * self.cell_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            if self.memory_cell_clip_value:
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output, -self.state_projection_clip_value, self.state_projection_clip_value)
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0:current_length_index + 1]
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output
        final_state = (full_batch_previous_state.unsqueeze(0), full_batch_previous_memory.unsqueeze(0))
        return (output_accumulator, final_state)