import numpy
import pytest
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.modules.span_extractors import BidirectionalEndpointSpanExtractor, SpanExtractor
from allennlp.nn.util import batched_index_select

class TestBidirectonalEndpointSpanExtractor:

    def test_bidirectional_endpoint_span_extractor_can_build_from_params(self):
        if False:
            i = 10
            return i + 15
        params = Params({'type': 'bidirectional_endpoint', 'input_dim': 4, 'num_width_embeddings': 5, 'span_width_embedding_dim': 3})
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, BidirectionalEndpointSpanExtractor)
        assert extractor.get_output_dim() == 2 + 2 + 3

    def test_raises_on_odd_input_dimension(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ConfigurationError):
            _ = BidirectionalEndpointSpanExtractor(7)

    def test_correct_sequence_elements_are_embedded(self):
        if False:
            while True:
                i = 10
        sequence_tensor = torch.randn([2, 5, 8])
        extractor = BidirectionalEndpointSpanExtractor(input_dim=8, forward_combination='x,y', backward_combination='x,y')
        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [3, 4]]])
        span_representations = extractor(sequence_tensor, indices)
        assert list(span_representations.size()) == [2, 2, 16]
        assert extractor.get_output_dim() == 16
        assert extractor.get_input_dim() == 8
        (forward_start_embeddings, forward_end_embeddings, backward_start_embeddings, backward_end_embeddings) = span_representations.split(4, -1)
        (forward_sequence_tensor, backward_sequence_tensor) = sequence_tensor.split(4, -1)
        correct_forward_start_indices = torch.LongTensor([[0, 1], [-1, 2]])
        correct_forward_start_indices[1, 0] = 1
        correct_forward_end_indices = torch.LongTensor([[3, 4], [2, 4]])
        correct_backward_start_indices = torch.LongTensor([[4, 5], [3, 5]])
        correct_backward_start_indices[0, 1] = 1
        correct_backward_start_indices[1, 1] = 1
        correct_backward_end_indices = torch.LongTensor([[1, 2], [0, 3]])
        correct_forward_start_embeddings = batched_index_select(forward_sequence_tensor.contiguous(), correct_forward_start_indices)
        correct_forward_start_embeddings[1, 0] = extractor._start_sentinel.data
        numpy.testing.assert_array_equal(forward_start_embeddings.data.numpy(), correct_forward_start_embeddings.data.numpy())
        correct_forward_end_embeddings = batched_index_select(forward_sequence_tensor.contiguous(), correct_forward_end_indices)
        numpy.testing.assert_array_equal(forward_end_embeddings.data.numpy(), correct_forward_end_embeddings.data.numpy())
        correct_backward_end_embeddings = batched_index_select(backward_sequence_tensor.contiguous(), correct_backward_end_indices)
        numpy.testing.assert_array_equal(backward_end_embeddings.data.numpy(), correct_backward_end_embeddings.data.numpy())
        correct_backward_start_embeddings = batched_index_select(backward_sequence_tensor.contiguous(), correct_backward_start_indices)
        correct_backward_start_embeddings[0, 1] = extractor._end_sentinel.data
        correct_backward_start_embeddings[1, 1] = extractor._end_sentinel.data
        numpy.testing.assert_array_equal(backward_start_embeddings.data.numpy(), correct_backward_start_embeddings.data.numpy())

    def test_correct_sequence_elements_are_embedded_with_a_masked_sequence(self):
        if False:
            i = 10
            return i + 15
        sequence_tensor = torch.randn([2, 5, 8])
        extractor = BidirectionalEndpointSpanExtractor(input_dim=8, forward_combination='x,y', backward_combination='x,y')
        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [0, 1]]])
        sequence_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        span_representations = extractor(sequence_tensor, indices, sequence_mask=sequence_mask)
        (forward_start_embeddings, forward_end_embeddings, backward_start_embeddings, backward_end_embeddings) = span_representations.split(4, -1)
        (forward_sequence_tensor, backward_sequence_tensor) = sequence_tensor.split(4, -1)
        correct_forward_start_indices = torch.LongTensor([[0, 1], [-1, -1]])
        correct_forward_start_indices[1, 0] = 1
        correct_forward_start_indices[1, 1] = 1
        correct_forward_end_indices = torch.LongTensor([[3, 4], [2, 1]])
        correct_backward_start_indices = torch.LongTensor([[4, 5], [3, 2]])
        correct_backward_start_indices[0, 1] = 1
        correct_backward_end_indices = torch.LongTensor([[1, 2], [0, 0]])
        correct_forward_start_embeddings = batched_index_select(forward_sequence_tensor.contiguous(), correct_forward_start_indices)
        correct_forward_start_embeddings[1, 0] = extractor._start_sentinel.data
        correct_forward_start_embeddings[1, 1] = extractor._start_sentinel.data
        numpy.testing.assert_array_equal(forward_start_embeddings.data.numpy(), correct_forward_start_embeddings.data.numpy())
        correct_forward_end_embeddings = batched_index_select(forward_sequence_tensor.contiguous(), correct_forward_end_indices)
        numpy.testing.assert_array_equal(forward_end_embeddings.data.numpy(), correct_forward_end_embeddings.data.numpy())
        correct_backward_end_embeddings = batched_index_select(backward_sequence_tensor.contiguous(), correct_backward_end_indices)
        numpy.testing.assert_array_equal(backward_end_embeddings.data.numpy(), correct_backward_end_embeddings.data.numpy())
        correct_backward_start_embeddings = batched_index_select(backward_sequence_tensor.contiguous(), correct_backward_start_indices)
        correct_backward_start_embeddings[0, 1] = extractor._end_sentinel.data
        correct_backward_start_embeddings[1, 0] = extractor._end_sentinel.data
        numpy.testing.assert_array_equal(backward_start_embeddings.data.numpy(), correct_backward_start_embeddings.data.numpy())

    def test_forward_doesnt_raise_with_empty_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        sequence_tensor = torch.FloatTensor([[[0.0, 0.0], [0.0, 0.0]]])
        sequence_mask = torch.tensor([[False, False]])
        span_indices = torch.LongTensor([[[-1, -1]]])
        span_indices_mask = torch.tensor([[False]])
        extractor = BidirectionalEndpointSpanExtractor(input_dim=2, forward_combination='x,y', backward_combination='x,y')
        span_representations = extractor(sequence_tensor, span_indices, sequence_mask=sequence_mask, span_indices_mask=span_indices_mask)
        numpy.testing.assert_array_equal(span_representations.detach(), torch.FloatTensor([[[0.0, 0.0, 0.0, 0.0]]]))

    def test_forward_raises_with_invalid_indices(self):
        if False:
            while True:
                i = 10
        sequence_tensor = torch.randn([2, 5, 8])
        extractor = BidirectionalEndpointSpanExtractor(input_dim=8)
        indices = torch.LongTensor([[[-1, 3], [7, 4]], [[0, 12], [0, -1]]])
        with pytest.raises(ValueError):
            _ = extractor(sequence_tensor, indices)