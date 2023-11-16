from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, batchsize=2):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("An invalid option for `--model` was supplied,\n                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']") from None
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.hidden = self.init_hidden(batchsize)

    @staticmethod
    def repackage_hidden(h):
        if False:
            while True:
                i = 10
        'Detach hidden states from their history.'
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple([RNNModel.repackage_hidden(v) for v in h])

    def init_weights(self):
        if False:
            while True:
                i = 10
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        if False:
            while True:
                i = 10
        emb = self.drop(self.encoder(input))
        (output, hidden) = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        self.hidden = RNNModel.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        if False:
            while True:
                i = 10
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(), weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()

class RNNModelWithTensorHidden(RNNModel):
    """Supports GRU scripting."""

    @staticmethod
    def repackage_hidden(h):
        if False:
            for i in range(10):
                print('nop')
        'Detach hidden states from their history.'
        return h.detach()

    def forward(self, input: Tensor, hidden: Tensor):
        if False:
            while True:
                i = 10
        emb = self.drop(self.encoder(input))
        (output, hidden) = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        self.hidden = RNNModelWithTensorHidden.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

class RNNModelWithTupleHidden(RNNModel):
    """Supports LSTM scripting."""

    @staticmethod
    def repackage_hidden(h: Tuple[Tensor, Tensor]):
        if False:
            i = 10
            return i + 15
        'Detach hidden states from their history.'
        return (h[0].detach(), h[1].detach())

    def forward(self, input: Tensor, hidden: Optional[Tuple[Tensor, Tensor]]=None):
        if False:
            while True:
                i = 10
        emb = self.drop(self.encoder(input))
        (output, hidden) = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        self.hidden = self.repackage_hidden(tuple(hidden))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))