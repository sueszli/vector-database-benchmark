import numpy
import pytest
import torch
from torch.nn import LSTM, RNN
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask

class TestEncoderBase(AllenNlpTestCase):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        super().setup_method()
        self.lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        self.rnn = RNN(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        self.encoder_base = _EncoderBase(stateful=True)
        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[3, 2:, :] = 0
        self.tensor = tensor
        mask = torch.ones(5, 7).bool()
        mask[1, 6:] = False
        mask[2, :] = False
        mask[3, 2:] = False
        mask[4, :] = False
        self.mask = mask
        self.batch_size = 5
        self.num_valid = 3
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        (_, _, restoration_indices, sorting_indices) = sort_batch_by_length(tensor, sequence_lengths)
        self.sorting_indices = sorting_indices
        self.restoration_indices = restoration_indices

    def test_non_stateful_states_are_sorted_correctly(self):
        if False:
            while True:
                i = 10
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (torch.randn(6, 5, 7), torch.randn(6, 5, 7))
        (_, states, restoration_indices) = encoder_base.sort_and_run_forward(lambda *x: x, self.tensor, self.mask, initial_states)
        zeros = torch.zeros([6, 2, 7])
        for (state, original) in zip(states, initial_states):
            assert list(state.size()) == [6, 3, 7]
            state_with_zeros = torch.cat([state, zeros], 1)
            unsorted_state = state_with_zeros.index_select(1, restoration_indices)
            for index in [0, 1, 3]:
                numpy.testing.assert_array_equal(unsorted_state[:, index, :].data.numpy(), original[:, index, :].data.numpy())

    def test_get_initial_states(self):
        if False:
            i = 10
            return i + 15
        assert self.encoder_base._get_initial_states(self.batch_size, self.num_valid, self.sorting_indices) is None
        initial_states = (torch.randn([1, 3, 7]), torch.randn([1, 3, 7]))
        self.encoder_base._states = initial_states
        returned_states = self.encoder_base._get_initial_states(self.batch_size, self.num_valid, self.sorting_indices)
        correct_expanded_states = [torch.cat([state, torch.zeros([1, 2, 7])], 1) for state in initial_states]
        numpy.testing.assert_array_equal(self.encoder_base._states[0].data.numpy(), correct_expanded_states[0].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1].data.numpy(), correct_expanded_states[1].data.numpy())
        correct_returned_states = [state.index_select(1, self.sorting_indices)[:, :self.num_valid, :] for state in correct_expanded_states]
        numpy.testing.assert_array_equal(returned_states[0].data.numpy(), correct_returned_states[0].data.numpy())
        numpy.testing.assert_array_equal(returned_states[1].data.numpy(), correct_returned_states[1].data.numpy())
        original_states = (torch.randn([1, 10, 7]), torch.randn([1, 10, 7]))
        self.encoder_base._states = original_states
        returned_states = self.encoder_base._get_initial_states(self.batch_size, self.num_valid, self.sorting_indices)
        numpy.testing.assert_array_equal(self.encoder_base._states[0].data.numpy(), original_states[0].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1].data.numpy(), original_states[1].data.numpy())
        correct_returned_state = [x.index_select(1, self.sorting_indices)[:, :self.num_valid, :] for x in original_states]
        numpy.testing.assert_array_equal(returned_states[0].data.numpy(), correct_returned_state[0].data.numpy())
        numpy.testing.assert_array_equal(returned_states[1].data.numpy(), correct_returned_state[1].data.numpy())

    def test_update_states(self):
        if False:
            while True:
                i = 10
        assert self.encoder_base._states is None
        initial_states = (torch.randn([1, 5, 7]), torch.randn([1, 5, 7]))
        index_selected_initial_states = (initial_states[0].index_select(1, self.restoration_indices), initial_states[1].index_select(1, self.restoration_indices))
        self.encoder_base._update_states(initial_states, self.restoration_indices)
        numpy.testing.assert_array_equal(self.encoder_base._states[0].data.numpy(), index_selected_initial_states[0].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1].data.numpy(), index_selected_initial_states[1].data.numpy())
        new_states = (torch.randn([1, 5, 7]), torch.randn([1, 5, 7]))
        new_states[0][:, -2:, :] = 0
        new_states[1][:, -2:, :] = 0
        index_selected_new_states = (new_states[0].index_select(1, self.restoration_indices), new_states[1].index_select(1, self.restoration_indices))
        self.encoder_base._update_states(new_states, self.restoration_indices)
        for index in [2, 4]:
            numpy.testing.assert_array_equal(self.encoder_base._states[0][:, index, :].data.numpy(), index_selected_initial_states[0][:, index, :].data.numpy())
            numpy.testing.assert_array_equal(self.encoder_base._states[1][:, index, :].data.numpy(), index_selected_initial_states[1][:, index, :].data.numpy())
        for index in [0, 1, 3]:
            numpy.testing.assert_array_equal(self.encoder_base._states[0][:, index, :].data.numpy(), index_selected_new_states[0][:, index, :].data.numpy())
            numpy.testing.assert_array_equal(self.encoder_base._states[1][:, index, :].data.numpy(), index_selected_new_states[1][:, index, :].data.numpy())
        small_new_states = (torch.randn([1, 3, 7]), torch.randn([1, 3, 7]))
        small_restoration_indices = torch.LongTensor([2, 0, 1])
        small_new_states[0][:, 0, :] = 0
        small_new_states[1][:, 0, :] = 0
        index_selected_small_states = (small_new_states[0].index_select(1, small_restoration_indices), small_new_states[1].index_select(1, small_restoration_indices))
        self.encoder_base._update_states(small_new_states, small_restoration_indices)
        for index in [1, 3]:
            numpy.testing.assert_array_equal(self.encoder_base._states[0][:, index, :].data.numpy(), index_selected_new_states[0][:, index, :].data.numpy())
            numpy.testing.assert_array_equal(self.encoder_base._states[1][:, index, :].data.numpy(), index_selected_new_states[1][:, index, :].data.numpy())
        for index in [0, 2]:
            numpy.testing.assert_array_equal(self.encoder_base._states[0][:, index, :].data.numpy(), index_selected_small_states[0][:, index, :].data.numpy())
            numpy.testing.assert_array_equal(self.encoder_base._states[1][:, index, :].data.numpy(), index_selected_small_states[1][:, index, :].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[0][:, 4, :].data.numpy(), index_selected_initial_states[0][:, 4, :].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1][:, 4, :].data.numpy(), index_selected_initial_states[1][:, 4, :].data.numpy())

    def test_reset_states(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.encoder_base._states is None
        initial_states = (torch.randn([1, 5, 7]), torch.randn([1, 5, 7]))
        index_selected_initial_states = (initial_states[0].index_select(1, self.restoration_indices), initial_states[1].index_select(1, self.restoration_indices))
        self.encoder_base._update_states(initial_states, self.restoration_indices)
        mask = torch.tensor([True, True, False, False, False])
        self.encoder_base.reset_states(mask)
        numpy.testing.assert_array_equal(self.encoder_base._states[0][:, :2, :].data.numpy(), torch.zeros_like(initial_states[0])[:, :2, :].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1][:, :2, :].data.numpy(), torch.zeros_like(initial_states[1])[:, :2, :].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[0][:, 2:, :].data.numpy(), index_selected_initial_states[0][:, 2:, :].data.numpy())
        numpy.testing.assert_array_equal(self.encoder_base._states[1][:, 2:, :].data.numpy(), index_selected_initial_states[1][:, 2:, :].data.numpy())
        bad_mask = torch.tensor([True, True, False])
        with pytest.raises(ValueError):
            self.encoder_base.reset_states(bad_mask)
        self.encoder_base.reset_states()
        assert self.encoder_base._states is None

    def test_non_contiguous_initial_states_handled(self):
        if False:
            while True:
                i = 10
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (torch.randn(5, 6, 7).permute(1, 0, 2), torch.randn(5, 6, 7).permute(1, 0, 2))
        assert not initial_states[0].is_contiguous() and (not initial_states[1].is_contiguous())
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask, initial_states)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask, initial_states[0])
        final_states = initial_states
        encoder_base = _EncoderBase(stateful=True)
        encoder_base._update_states(final_states, self.restoration_indices)
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask)
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]], self.restoration_indices)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask)

    @requires_gpu
    def test_non_contiguous_initial_states_handled_on_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        encoder_base = _EncoderBase(stateful=False).cuda()
        initial_states = (torch.randn(5, 6, 7).cuda().permute(1, 0, 2), torch.randn(5, 6, 7).cuda().permute(1, 0, 2))
        assert not initial_states[0].is_contiguous() and (not initial_states[1].is_contiguous())
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])
        encoder_base.sort_and_run_forward(self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda(), initial_states)
        encoder_base.sort_and_run_forward(self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda(), initial_states[0])
        final_states = initial_states
        encoder_base = _EncoderBase(stateful=True).cuda()
        encoder_base._update_states(final_states, self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda())
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]], self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda())