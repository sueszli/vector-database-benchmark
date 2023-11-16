import torch.nn as nn
import torch.nn.functional as F
from fairseq.data import Dictionary
from fairseq.models import FairseqDecoder, FairseqLanguageModel, register_model, register_model_architecture

@register_model('dummy_model')
class DummyModel(FairseqLanguageModel):

    def __init__(self, args, encoder):
        if False:
            i = 10
            return i + 15
        super().__init__(encoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('--num-layers', type=int, default=24)
        parser.add_argument('--embed-dim', type=int, default=1024)

    @classmethod
    def build_model(cls, args, task):
        if False:
            for i in range(10):
                print('nop')
        encoder = DummyEncoder(num_embed=len(task.target_dictionary), embed_dim=args.embed_dim, num_layers=args.num_layers)
        return cls(args, encoder)

    def forward(self, src_tokens, masked_tokens=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder(src_tokens, masked_tokens=masked_tokens)

class DummyEncoder(FairseqDecoder):

    def __init__(self, num_embed=50000, embed_dim=1024, num_layers=24):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(Dictionary())
        self.embed = nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim, padding_idx=0)
        self.layers_a = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3 * embed_dim), nn.Linear(3 * embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Dropout()) for i in range(num_layers)])
        self.layers_b = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(0.1)) for i in range(num_layers)])
        self.out_proj = nn.Linear(embed_dim, num_embed)

    def forward(self, tokens, masked_tokens=None):
        if False:
            while True:
                i = 10
        x = self.embed(tokens)
        for (layer_a, layer_b) in zip(self.layers_a, self.layers_b):
            x = x + layer_a(x)
            x = x + layer_b(x)
        x = self.out_proj(x)
        if masked_tokens is not None:
            x = x[masked_tokens]
        return (x,)

    def max_positions(self):
        if False:
            for i in range(10):
                print('nop')
        return 1024

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            for i in range(10):
                print('nop')
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

@register_model_architecture('dummy_model', 'dummy_model')
def base_architecture(args):
    if False:
        while True:
            i = 10
    pass