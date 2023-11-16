from __future__ import annotations
from ...data.all import *
from ..core import *
__all__ = ['awd_lstm_lm_config', 'awd_lstm_clas_config', 'dropout_mask', 'RNNDropout', 'WeightDropout', 'EmbeddingDropout', 'AWD_LSTM', 'awd_lstm_lm_split', 'awd_lstm_clas_split']

def dropout_mask(x: Tensor, sz: list, p: float) -> Tensor:
    if False:
        while True:
            i = 10
    'Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.'
    return x.new_empty(*sz).bernoulli_(1 - p).div_(1 - p)

class RNNDropout(Module):
    """Dropout with probability `p` that is consistent on the seq_len dimension."""

    def __init__(self, p: float=0.5):
        if False:
            print('Hello World!')
        self.p = p

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not self.training or self.p == 0.0:
            return x
        return x * dropout_mask(x.data, (x.size(0), 1, *x.shape[2:]), self.p)

class WeightDropout(Module):
    """A module that wraps another layer in which some weights will be replaced by 0 during training."""

    def __init__(self, module: nn.Module, weight_p: float, layer_names: str | MutableSequence='weight_hh_l0'):
        if False:
            return 10
        (self.module, self.weight_p, self.layer_names) = (module, weight_p, L(layer_names))
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            setattr(self.module, layer, w.clone())
            if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
                self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        if False:
            while True:
                i = 10
        'Apply dropout to the raw weights.'
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            if self.training:
                w = F.dropout(raw_w, p=self.weight_p)
            else:
                w = raw_w.clone()
            setattr(self.module, layer, w)

    def forward(self, *args):
        if False:
            while True:
                i = 10
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            return self.module(*args)

    def reset(self):
        if False:
            while True:
                i = 10
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            setattr(self.module, layer, raw_w.clone())
        if hasattr(self.module, 'reset'):
            self.module.reset()

    def _do_nothing(self):
        if False:
            print('Hello World!')
        pass

class EmbeddingDropout(Module):
    """Apply dropout with probability `embed_p` to an embedding layer `emb`."""

    def __init__(self, emb: nn.Embedding, embed_p: float):
        if False:
            i = 10
            return i + 15
        (self.emb, self.embed_p) = (emb, embed_p)

    def forward(self, words, scale=None):
        if False:
            return 10
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, ifnone(self.emb.padding_idx, -1), self.emb.max_norm, self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

class AWD_LSTM(Module):
    """AWD-LSTM inspired by https://arxiv.org/abs/1708.02182"""
    initrange = 0.1

    def __init__(self, vocab_sz: int, emb_sz: int, n_hid: int, n_layers: int, pad_token: int=1, hidden_p: float=0.2, input_p: float=0.6, embed_p: float=0.1, weight_p: float=0.5, bidir: bool=False):
        if False:
            i = 10
            return i + 15
        store_attr('emb_sz,n_hid,n_layers,pad_token')
        self.bs = 1
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.ModuleList([self._one_rnn(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz) // self.n_dir, bidir, weight_p, l) for l in range(n_layers)])
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.reset()

    def forward(self, inp: Tensor, from_embeds: bool=False):
        if False:
            for i in range(10):
                print('nop')
        (bs, sl) = inp.shape[:2] if from_embeds else inp.shape
        if bs != self.bs:
            self._change_hidden(bs)
        output = self.input_dp(inp if from_embeds else self.encoder_dp(inp))
        new_hidden = []
        for (l, (rnn, hid_dp)) in enumerate(zip(self.rnns, self.hidden_dps)):
            (output, new_h) = rnn(output, self.hidden[l])
            new_hidden.append(new_h)
            if l != self.n_layers - 1:
                output = hid_dp(output)
        self.hidden = to_detach(new_hidden, cpu=False, gather=False)
        return output

    def _change_hidden(self, bs):
        if False:
            print('Hello World!')
        self.hidden = [self._change_one_hidden(l, bs) for l in range(self.n_layers)]
        self.bs = bs

    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        if False:
            print('Hello World!')
        'Return one of the inner rnn'
        rnn = nn.LSTM(n_in, n_out, 1, batch_first=True, bidirectional=bidir)
        return WeightDropout(rnn, weight_p)

    def _one_hidden(self, l):
        if False:
            i = 10
            return i + 15
        'Return one hidden state'
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return (one_param(self).new_zeros(self.n_dir, self.bs, nh), one_param(self).new_zeros(self.n_dir, self.bs, nh))

    def _change_one_hidden(self, l, bs):
        if False:
            return 10
        if self.bs < bs:
            nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            return tuple((torch.cat([h, h.new_zeros(self.n_dir, bs - self.bs, nh)], dim=1) for h in self.hidden[l]))
        if self.bs > bs:
            return (self.hidden[l][0][:, :bs].contiguous(), self.hidden[l][1][:, :bs].contiguous())
        return self.hidden[l]

    def reset(self):
        if False:
            i = 10
            return i + 15
        'Reset the hidden states'
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]

def awd_lstm_lm_split(model):
    if False:
        i = 10
        return i + 15
    'Split a RNN `model` in groups for differential learning rates.'
    groups = [nn.Sequential(rnn, dp) for (rnn, dp) in zip(model[0].rnns, model[0].hidden_dps)]
    groups = L(groups + [nn.Sequential(model[0].encoder, model[0].encoder_dp, model[1])])
    return groups.map(params)
awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1, hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

def awd_lstm_clas_split(model):
    if False:
        while True:
            i = 10
    'Split a RNN `model` in groups for differential learning rates.'
    groups = [nn.Sequential(model[0].module.encoder, model[0].module.encoder_dp)]
    groups += [nn.Sequential(rnn, dp) for (rnn, dp) in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    groups = L(groups + [model[1]])
    return groups.map(params)
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4, hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)