import numpy
import torch
from allennlp.modules.span_extractors import SpanExtractor, SelfAttentiveSpanExtractor
from allennlp.common.params import Params

class TestSelfAttentiveSpanExtractor:

    def test_locally_normalised_span_extractor_can_build_from_params(self):
        if False:
            return 10
        params = Params({'type': 'self_attentive', 'input_dim': 7, 'num_width_embeddings': 5, 'span_width_embedding_dim': 3})
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, SelfAttentiveSpanExtractor)
        assert extractor.get_output_dim() == 10

    def test_attention_is_normalised_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        input_dim = 7
        sequence_tensor = torch.randn([2, 5, input_dim])
        extractor = SelfAttentiveSpanExtractor(input_dim=input_dim)
        assert extractor.get_output_dim() == input_dim
        assert extractor.get_input_dim() == input_dim
        extractor._global_attention._module.weight.data.fill_(0.0)
        extractor._global_attention._module.bias.data.fill_(0.0)
        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [3, 4]]])
        span_representations = extractor(sequence_tensor, indices)
        assert list(span_representations.size()) == [2, 2, input_dim]
        batch_element = 0
        spans = span_representations[batch_element]
        mean_embeddings = sequence_tensor[batch_element, 1:4, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        mean_embeddings = sequence_tensor[batch_element, 2:5, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), mean_embeddings.data.numpy())
        batch_element = 1
        spans = span_representations[batch_element]
        mean_embeddings = sequence_tensor[batch_element, 0:3, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        mean_embeddings = sequence_tensor[batch_element, 3:5, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), mean_embeddings.data.numpy())
        indices_mask = torch.tensor([[True, True], [True, False]])
        span_representations = extractor(sequence_tensor, indices, span_indices_mask=indices_mask)
        batch_element = 0
        spans = span_representations[batch_element]
        mean_embeddings = sequence_tensor[batch_element, 1:4, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        mean_embeddings = sequence_tensor[batch_element, 2:5, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), mean_embeddings.data.numpy())
        batch_element = 1
        spans = span_representations[batch_element]
        mean_embeddings = sequence_tensor[batch_element, 0:3, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), numpy.zeros([input_dim]))

    def test_widths_are_embedded_correctly(self):
        if False:
            i = 10
            return i + 15
        input_dim = 7
        max_span_width = 5
        span_width_embedding_dim = 3
        output_dim = input_dim + span_width_embedding_dim
        extractor = SelfAttentiveSpanExtractor(input_dim=input_dim, num_width_embeddings=max_span_width, span_width_embedding_dim=span_width_embedding_dim)
        assert extractor.get_output_dim() == output_dim
        assert extractor.get_input_dim() == input_dim
        sequence_tensor = torch.randn([2, max_span_width, input_dim])
        indices = torch.LongTensor([[[1, 3], [0, 4], [0, 0]], [[0, 2], [1, 4], [2, 2]]])
        span_representations = extractor(sequence_tensor, indices)
        assert list(span_representations.size()) == [2, 3, output_dim]
        width_embeddings = extractor._span_width_embedding.weight.data.numpy()
        widths_minus_one = indices[..., 1] - indices[..., 0]
        for element in range(indices.size(0)):
            for span in range(indices.size(1)):
                width = widths_minus_one[element, span].item()
                width_embedding = span_representations[element, span, input_dim:]
                numpy.testing.assert_array_almost_equal(width_embedding.data.numpy(), width_embeddings[width])