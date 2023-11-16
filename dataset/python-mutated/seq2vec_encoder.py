from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable

class Seq2VecEncoder(_EncoderBase, Registrable):
    """
    A `Seq2VecEncoder` is a `Module` that takes as input a sequence of vectors and returns a
    single vector.  Input shape : `(batch_size, sequence_length, input_dim)`; output shape:
    `(batch_size, output_dim)`.

    We add two methods to the basic `Module` API: `get_input_dim()` and `get_output_dim()`.
    You might need this if you want to construct a `Linear` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """

    def get_input_dim(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the dimension of the vector input for each element in the sequence input\n        to a `Seq2VecEncoder`. This is `not` the shape of the input tensor, but the\n        last element of that shape.\n        '
        raise NotImplementedError

    def get_output_dim(self) -> int:
        if False:
            return 10
        '\n        Returns the dimension of the final vector output by this `Seq2VecEncoder`.  This is `not`\n        the shape of the returned tensor, but the last element of that shape.\n        '
        raise NotImplementedError