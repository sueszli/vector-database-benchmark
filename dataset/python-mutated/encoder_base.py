from typing import Tuple, Union, Optional, Callable, Any
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, sort_batch_by_length
RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]

class _EncoderBase(torch.nn.Module):
    """
    This abstract class serves as a base for the 3 `Encoder` abstractions in AllenNLP.
    - [`Seq2SeqEncoders`](./seq2seq_encoders/seq2seq_encoder.md)
    - [`Seq2VecEncoders`](./seq2vec_encoders/seq2vec_encoder.md)

    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool=False) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence, Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor], RnnState]], inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: Optional[RnnState]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function exists because Pytorch RNNs require that their inputs be sorted\n        before being passed as input. As all of our Seq2xxxEncoders use this functionality,\n        it is provided in a base class. This method can be called on any module which\n        takes as input a `PackedSequence` and some `hidden_state`, which can either be a\n        tuple of tensors or a tensor.\n\n        As all of our Seq2xxxEncoders have different return types, we return `sorted`\n        outputs from the module, which is called directly. Additionally, we return the\n        indices into the batch dimension required to restore the tensor to it's correct,\n        unsorted order and the number of valid batch elements (i.e the number of elements\n        in the batch which are not completely masked). This un-sorting and re-padding\n        of the module outputs is left to the subclasses because their outputs have different\n        types and handling them smoothly here is difficult.\n\n        # Parameters\n\n        module : `Callable[RnnInputs, RnnOutputs]`\n            A function to run on the inputs, where\n            `RnnInputs: [PackedSequence, Optional[RnnState]]` and\n            `RnnOutputs: Tuple[Union[PackedSequence, torch.Tensor], RnnState]`.\n            In most cases, this is a `torch.nn.Module`.\n        inputs : `torch.Tensor`, required.\n            A tensor of shape `(batch_size, sequence_length, embedding_size)` representing\n            the inputs to the Encoder.\n        mask : `torch.BoolTensor`, required.\n            A tensor of shape `(batch_size, sequence_length)`, representing masked and\n            non-masked elements of the sequence for each element in the batch.\n        hidden_state : `Optional[RnnState]`, (default = `None`).\n            A single tensor of shape (num_layers, batch_size, hidden_size) representing the\n            state of an RNN with or a tuple of\n            tensors of shapes (num_layers, batch_size, hidden_size) and\n            (num_layers, batch_size, memory_size), representing the hidden state and memory\n            state of an LSTM-like RNN.\n\n        # Returns\n\n        module_output : `Union[torch.Tensor, PackedSequence]`.\n            A Tensor or PackedSequence representing the output of the Pytorch Module.\n            The batch size dimension will be equal to `num_valid`, as sequences of zero\n            length are clipped off before the module is called, as Pytorch cannot handle\n            zero length sequences.\n        final_states : `Optional[RnnState]`\n            A Tensor representing the hidden state of the Pytorch Module. This can either\n            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in\n            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.\n        restoration_indices : `torch.LongTensor`\n            A tensor of shape `(batch_size,)`, describing the re-indexing required to transform\n            the outputs back to their original batch order.\n        "
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        (sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices) = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :], sorted_sequence_lengths[:num_valid].data.tolist(), batch_first=True)
        if not self.stateful:
            if hidden_state is None:
                initial_states: Any = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous() for state in hidden_state]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)
        (module_output, final_states) = module(packed_sequence_input, initial_states)
        return (module_output, final_states, restoration_indices)

    def _get_initial_states(self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor) -> Optional[RnnState]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an initial state for use in an RNN. Additionally, this method handles\n        the batch size changing across calls by mutating the state to append initial states\n        for new elements in the batch. Finally, it also handles sorting the states\n        with respect to the sequence lengths of elements in the batch and removing rows\n        which are completely padded. Importantly, this `mutates` the state if the\n        current batch size is larger than when it was previously called.\n\n        # Parameters\n\n        batch_size : `int`, required.\n            The batch size can change size across calls to stateful RNNs, so we need\n            to know if we need to expand or shrink the states before returning them.\n            Expanded states will be set to zero.\n        num_valid : `int`, required.\n            The batch may contain completely padded sequences which get removed before\n            the sequence is passed through the encoder. We also need to clip these off\n            of the state too.\n        sorting_indices `torch.LongTensor`, required.\n            Pytorch RNNs take sequences sorted by length. When we return the states to be\n            used for a given call to `module.forward`, we need the states to match up to\n            the sorted sequences, so before returning them, we sort the states using the\n            same indices used to sort the sequences.\n\n        # Returns\n\n        This method has a complex return type because it has to deal with the first time it\n        is called, when it has no state, and the fact that types of RNN have heterogeneous\n        states.\n\n        If it is the first time the module has been called, it returns `None`, regardless\n        of the type of the `Module`.\n\n        Otherwise, for LSTMs, it returns a tuple of `torch.Tensors` with shape\n        `(num_layers, num_valid, state_size)` and `(num_layers, num_valid, memory_size)`\n        respectively, or for GRUs, it returns a single `torch.Tensor` of shape\n        `(num_layers, num_valid, state_size)`.\n        '
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = state.new_zeros(state.size(0), num_states_to_concat, state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple((state[:, :batch_size, :] for state in self._states))
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :].contiguous()
        else:
            sorted_states = [state.index_select(1, sorting_indices) for state in correctly_shaped_states]
            return tuple((state[:, :num_valid, :].contiguous() for state in sorted_states))

    def _update_states(self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor) -> None:
        if False:
            while True:
                i = 10
        '\n        After the RNN has run forward, the states need to be updated.\n        This method just sets the state to the updated new state, performing\n        several pieces of book-keeping along the way - namely, unsorting the\n        states and ensuring that the states of completely padded sequences are\n        not updated. Finally, it also detaches the state variable from the\n        computational graph, such that the graph can be garbage collected after\n        each batch iteration.\n\n        # Parameters\n\n        final_states : `RnnStateStorage`, required.\n            The hidden states returned as output from the RNN.\n        restoration_indices : `torch.LongTensor`, required.\n            The indices that invert the sorting used in `sort_and_run_forward`\n            to order the states with respect to the lengths of the sequences in\n            the batch.\n        '
        new_unsorted_states = [state.index_select(1, restoration_indices) for state in final_states]
        if self._states is None:
            self._states = tuple((state.data for state in new_unsorted_states))
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [(state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1) for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for (old_state, new_state, used_mask) in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for (old_state, new_state, used_mask) in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())
            self._states = tuple(new_states)

    def reset_states(self, mask: torch.BoolTensor=None) -> None:
        if False:
            return 10
        '\n        Resets the internal states of a stateful encoder.\n\n        # Parameters\n\n        mask : `torch.BoolTensor`, optional.\n            A tensor of shape `(batch_size,)` indicating which states should\n            be reset. If not provided, all states will be reset.\n        '
        if mask is None:
            self._states = None
        else:
            mask_batch_size = mask.size(0)
            mask = mask.view(1, mask_batch_size, 1)
            new_states = []
            assert self._states is not None
            for old_state in self._states:
                old_state_batch_size = old_state.size(1)
                if old_state_batch_size != mask_batch_size:
                    raise ValueError(f'Trying to reset states using mask with incorrect batch size. Expected batch size: {old_state_batch_size}. Provided batch size: {mask_batch_size}.')
                new_state = ~mask * old_state
                new_states.append(new_state.detach())
            self._states = tuple(new_states)