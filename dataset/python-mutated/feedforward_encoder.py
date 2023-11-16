import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register('feedforward')
class FeedForwardEncoder(Seq2SeqEncoder):
    """
    This class applies the `FeedForward` to each item in sequences.

    Registered as a `Seq2SeqEncoder` with name "feedforward".
    """

    def __init__(self, feedforward: FeedForward) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._feedforward = feedforward

    def get_input_dim(self) -> int:
        if False:
            while True:
                i = 10
        return self._feedforward.get_input_dim()

    def get_output_dim(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._feedforward.get_output_dim()

    def is_bidirectional(self) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        # Parameters\n\n        inputs : `torch.Tensor`, required.\n            A tensor of shape (batch_size, timesteps, input_dim)\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A tensor of shape (batch_size, timesteps).\n\n        # Returns\n\n        A tensor of shape (batch_size, timesteps, output_dim).\n        '
        if mask is None:
            return self._feedforward(inputs)
        else:
            outputs = self._feedforward(inputs)
            return outputs * mask.unsqueeze(dim=-1)