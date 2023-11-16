import torch
import torch.nn.functional as F

class MeanPoolGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def forward(self, encoder_out):
        if False:
            print('Hello World!')
        if not ('encoder_out' in encoder_out and 'encoder_padding_mask' in encoder_out and (encoder_out['encoder_out'][0].size(2) == self.embed_dim)):
            raise ValueError('Unexpected format for encoder_out')
        encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
        encoder_out = encoder_out['encoder_out'][0].transpose(0, 1)
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)
        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)