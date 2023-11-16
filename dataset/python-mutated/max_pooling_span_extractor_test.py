import pytest
import torch
from allennlp.common.params import Params
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors.max_pooling_span_extractor import MaxPoolingSpanExtractor

class TestMaxPoolingSpanExtractor:

    def test_locally_span_extractor_can_build_from_params(self):
        if False:
            while True:
                i = 10
        params = Params({'type': 'max_pooling', 'input_dim': 3, 'num_width_embeddings': 5, 'span_width_embedding_dim': 3})
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, MaxPoolingSpanExtractor)
        assert extractor.get_output_dim() == 6

    def test_max_values_extracted(self):
        if False:
            print('Hello World!')
        sequence_tensor = torch.randn([2, 10, 30])
        extractor = MaxPoolingSpanExtractor(30)
        indices = torch.LongTensor([[[1, 1], [2, 4], [9, 9]], [[0, 1], [4, 4], [0, 9]]])
        span_representations = extractor(sequence_tensor, indices)
        assert list(span_representations.size()) == [2, 3, 30]
        assert extractor.get_output_dim() == 30
        assert extractor.get_input_dim() == 30
        for (batch, X) in enumerate(indices):
            for (indices_ind, span_def) in enumerate(X):
                span_features_complete = sequence_tensor[batch][span_def[0]:span_def[1] + 1]
                for i in range(extractor.get_output_dim()):
                    features_from_span = span_features_complete[:, i]
                    real_max_value = max(features_from_span)
                    extracted_max_value = span_representations[batch, indices_ind, i]
                    assert real_max_value == extracted_max_value, f'Error extracting max value for batch {batch}, span {indices_ind} on dimension {i}.expected {real_max_value} but got {extracted_max_value} which is not the maximum element.'

    def test_sequence_mask_correct_excluded(self):
        if False:
            i = 10
            return i + 15
        sequence_tensor = torch.randn([2, 6, 30])
        extractor = MaxPoolingSpanExtractor(30)
        indices = torch.LongTensor([[[1, 1], [3, 5], [2, 5]], [[0, 0], [0, 3], [4, 5]]])
        seq_mask = torch.BoolTensor([[True] * 4 + [False] * 2, [True] * 5 + [False] * 1])
        span_representations = extractor(sequence_tensor, indices, sequence_mask=seq_mask)
        sequence_tensor[torch.logical_not(seq_mask)] = float('-inf')
        for (batch, X) in enumerate(indices):
            for (indices_ind, span_def) in enumerate(X):
                span_features_complete = sequence_tensor[batch][span_def[0]:span_def[1] + 1]
                for (i, _) in enumerate(span_features_complete):
                    features_from_span = span_features_complete[:, i]
                    real_max_value = max(features_from_span)
                    extracted_max_value = span_representations[batch, indices_ind, i]
                    assert real_max_value == extracted_max_value, f'Error extracting max value for batch {batch}, span {indices_ind} on dimension {i}.expected {real_max_value} but got {extracted_max_value} which is not the maximum element.'

    def test_span_mask_correct_excluded(self):
        if False:
            return 10
        sequence_tensor = torch.randn([2, 6, 10])
        extractor = MaxPoolingSpanExtractor(10)
        indices = torch.LongTensor([[[1, 1], [3, 5], [2, 5]], [[0, 0], [0, 3], [4, 5]]])
        span_mask = torch.BoolTensor([[True] * 3, [False] * 3])
        span_representations = extractor(sequence_tensor, indices, span_indices_mask=span_mask)
        X = indices[-1]
        batch = -1
        for (indices_ind, span_def) in enumerate(X):
            span_features_complete = sequence_tensor[batch][span_def[0]:span_def[1] + 1]
            for (i, _) in enumerate(span_features_complete):
                real_max_value = torch.FloatTensor([0.0])
                extracted_max_value = span_representations[batch, indices_ind, i]
                assert real_max_value == extracted_max_value, f'Error extracting max value for batch {batch}, span {indices_ind} on dimension {i}.expected {real_max_value} but got {extracted_max_value} which is not the maximum element.'

    def test_inconsistent_extractor_dimension_throws_exception(self):
        if False:
            return 10
        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[1, 1], [2, 4], [9, 9]], [[0, 1], [4, 4], [0, 9]]])
        with pytest.raises(ValueError):
            extractor = MaxPoolingSpanExtractor(9)
            extractor(sequence_tensor, indices)
        with pytest.raises(ValueError):
            extractor = MaxPoolingSpanExtractor(11)
            extractor(sequence_tensor, indices)

    def test_span_indices_outside_sequence(self):
        if False:
            print('Hello World!')
        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[6, 6], [2, 4]], [[0, 1], [4, 4]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)
        indices = torch.LongTensor([[[5, 6], [2, 4]], [[0, 1], [4, 4]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)
        indices = torch.LongTensor([[[-1, 0], [2, 4]], [[0, 1], [4, 4]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

    def test_span_start_below_span_end(self):
        if False:
            i = 10
            return i + 15
        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[4, 2], [2, 4], [1, 1]], [[0, 1], [4, 4], [1, 1]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

    def test_span_sequence_complete_masked(self):
        if False:
            return 10
        sequence_tensor = torch.randn([2, 6, 10])
        seq_mask = torch.BoolTensor([[True] * 2 + [False] * 4, [True] * 3 + [False] * 3])
        indices = torch.LongTensor([[[5, 5]], [[4, 5]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices, sequence_mask=seq_mask)