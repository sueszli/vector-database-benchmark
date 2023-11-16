import torch
from allennlp.common.registrable import Registrable

class SpanExtractor(torch.nn.Module, Registrable):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def forward(self, sequence_tensor: torch.FloatTensor, span_indices: torch.LongTensor, sequence_mask: torch.BoolTensor=None, span_indices_mask: torch.BoolTensor=None):
        if False:
            print('Hello World!')
        "\n        Given a sequence tensor, extract spans and return representations of\n        them. Span representation can be computed in many different ways,\n        such as concatenation of the start and end spans, attention over the\n        vectors contained inside the span, etc.\n\n        # Parameters\n\n        sequence_tensor : `torch.FloatTensor`, required.\n            A tensor of shape (batch_size, sequence_length, embedding_size)\n            representing an embedded sequence of words.\n        span_indices : `torch.LongTensor`, required.\n            A tensor of shape `(batch_size, num_spans, 2)`, where the last\n            dimension represents the inclusive start and end indices of the\n            span to be extracted from the `sequence_tensor`.\n        sequence_mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of shape (batch_size, sequence_length) representing padded\n            elements of the sequence.\n        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of shape (batch_size, num_spans) representing the valid\n            spans in the `indices` tensor. This mask is optional because\n            sometimes it's easier to worry about masking after calling this\n            function, rather than passing a mask directly.\n\n        # Returns\n\n        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,\n        where `embedded_span_size` depends on the way spans are represented.\n        "
        raise NotImplementedError

    def get_input_dim(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the expected final dimension of the `sequence_tensor`.\n        '
        raise NotImplementedError

    def get_output_dim(self) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the expected final dimension of the returned span representation.\n        '
        raise NotImplementedError