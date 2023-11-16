from torch import nn
from fairseq.models import FairseqEncoder

class CTCDecoder(FairseqEncoder):

    def __init__(self, dictionary, in_dim):
        if False:
            i = 10
            return i + 15
        super().__init__(dictionary)
        self.proj = nn.Linear(in_dim, len(dictionary))

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        if False:
            while True:
                i = 10
        encoder_out = self.proj(src_tokens)
        return {'encoder_out': encoder_out}