import numpy as np
import functools
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
DTYPE = torch.bool

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """

    def __init__(self, batch_idxs_np, device):
        if False:
            i = 10
            return i + 15
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = torch.as_tensor(batch_idxs_np, dtype=torch.long, device=device)
        self.batch_size = int(1 + np.max(batch_idxs_np))
        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))

class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        if ctx.p > 0 and ctx.train:
            return (grad_output.mul(ctx.noise), None, None, None, None)
        else:
            return (grad_output, None, None, None, None)

class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def __init__(self, p=0.5, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        if False:
            i = 10
            return i + 15
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

class LayerNormalization(nn.Module):

    def __init__(self, d_hid, eps=0.001, affine=True):
        if False:
            while True:
                i = 10
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if False:
            i = 10
            return i + 15
        if z.size(-1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attention_dropout=0.1):
        if False:
            return 10
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        if False:
            while True:
                i = 10
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return (output, attn)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        if False:
            while True:
                i = 10
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        if not d_positional:
            self.partitioned = False
        else:
            self.partitioned = True
        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional
            self.w_qs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_v // 2))
            self.w_qs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_v // 2))
            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)
            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))
            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)
        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)
        if not self.partitioned:
            self.proj = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head * (d_v // 2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head * (d_v // 2), self.d_positional, bias=False)
        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        if False:
            while True:
                i = 10
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1))
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))
        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs)
            k_s = torch.bmm(qk_inp_repeated, self.w_ks)
            v_s = torch.bmm(v_inp_repeated, self.w_vs)
        else:
            q_s = torch.cat([torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_qs1), torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_qs2)], -1)
            k_s = torch.cat([torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_ks1), torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_ks2)], -1)
            v_s = torch.cat([torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1), torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2)], -1)
        return (q_s, k_s, v_s)

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        if False:
            while True:
                i = 10
        n_head = self.n_head
        (d_k, d_v) = (self.d_k, self.d_v)
        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)
        for (i, (start, end)) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:, i, :end - start, :] = q_s[:, start:end, :]
            k_padded[:, i, :end - start, :] = k_s[:, start:end, :]
            v_padded[:, i, :end - start, :] = v_s[:, start:end, :]
            invalid_mask[i, :end - start].fill_(False)
        return (q_padded.view(-1, len_padded, d_k), k_padded.view(-1, len_padded, d_k), v_padded.view(-1, len_padded, d_v), invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1), (~invalid_mask).repeat(n_head, 1))

    def combine_v(self, outputs):
        if False:
            print('Hello World!')
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)
        if not self.partitioned:
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([self.proj1(outputs1), self.proj2(outputs2)], -1)
        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        if False:
            print('Hello World!')
        residual = inp
        (q_s, k_s, v_s) = self.split_qkv_packed(inp, qk_inp=qk_inp)
        (q_padded, k_padded, v_padded, attn_mask, output_mask) = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        (outputs_padded, attns_padded) = self.attention(q_padded, k_padded, v_padded, attn_mask=attn_mask)
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)
        outputs = self.residual_dropout(outputs, batch_idxs)
        return (self.layer_norm(outputs + residual), attns_padded)

class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        if False:
            return 10
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        if False:
            while True:
                i = 10
        residual = x
        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)
        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

class PartitionedPositionwiseFeedForward(nn.Module):

    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff // 2)
        self.w_1p = nn.Linear(d_positional, d_ff // 2)
        self.w_2c = nn.Linear(d_ff // 2, self.d_content)
        self.w_2p = nn.Linear(d_ff // 2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        if False:
            return 10
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]
        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)
        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)
        output = torch.cat([outputc, outputp], -1)
        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)

class LabelAttention(nn.Module):
    """
    Single-head Attention layer for label-specific representations
    """

    def __init__(self, d_model, d_k, d_v, d_l, d_proj, combine_as_self, use_resdrop=True, q_as_matrix=False, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        if False:
            i = 10
            return i + 15
        super(LabelAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_l = d_l
        self.d_model = d_model
        self.d_proj = d_proj
        self.use_resdrop = use_resdrop
        self.q_as_matrix = q_as_matrix
        self.combine_as_self = combine_as_self
        if not d_positional:
            self.partitioned = False
        else:
            self.partitioned = True
        if self.partitioned:
            if d_model <= d_positional:
                raise ValueError('Unable to build LabelAttention.  d_model %d <= d_positional %d' % (d_model, d_positional))
            self.d_content = d_model - d_positional
            self.d_positional = d_positional
            if self.q_as_matrix:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            else:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            self.w_vs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_v // 2), requires_grad=True)
            if self.q_as_matrix:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            else:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            self.w_vs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_v // 2), requires_grad=True)
            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)
            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            if self.q_as_matrix:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            else:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_k), requires_grad=True)
            self.w_ks = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            self.w_vs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)
            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)
        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        if self.combine_as_self:
            self.layer_norm = LayerNormalization(d_model)
        else:
            self.layer_norm = LayerNormalization(self.d_proj)
        if not self.partitioned:
            if self.combine_as_self:
                self.proj = nn.Linear(self.d_l * d_v, d_model, bias=False)
            else:
                self.proj = nn.Linear(d_v, d_model, bias=False)
        elif self.combine_as_self:
            self.proj1 = nn.Linear(self.d_l * (d_v // 2), self.d_content, bias=False)
            self.proj2 = nn.Linear(self.d_l * (d_v // 2), self.d_positional, bias=False)
        else:
            self.proj1 = nn.Linear(d_v // 2, self.d_content, bias=False)
            self.proj2 = nn.Linear(d_v // 2, self.d_positional, bias=False)
        if not self.combine_as_self:
            self.reduce_proj = nn.Linear(d_model, self.d_proj, bias=False)
        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, k_inp=None):
        if False:
            return 10
        len_inp = inp.size(0)
        v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1))
        if k_inp is None:
            k_inp_repeated = v_inp_repeated
        else:
            k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1))
        if not self.partitioned:
            if self.q_as_matrix:
                q_s = torch.bmm(k_inp_repeated, self.w_qs)
            else:
                q_s = self.w_qs.unsqueeze(1)
            k_s = torch.bmm(k_inp_repeated, self.w_ks)
            v_s = torch.bmm(v_inp_repeated, self.w_vs)
        else:
            if self.q_as_matrix:
                q_s = torch.cat([torch.bmm(k_inp_repeated[:, :, :self.d_content], self.w_qs1), torch.bmm(k_inp_repeated[:, :, self.d_content:], self.w_qs2)], -1)
            else:
                q_s = torch.cat([self.w_qs1.unsqueeze(1), self.w_qs2.unsqueeze(1)], -1)
            k_s = torch.cat([torch.bmm(k_inp_repeated[:, :, :self.d_content], self.w_ks1), torch.bmm(k_inp_repeated[:, :, self.d_content:], self.w_ks2)], -1)
            v_s = torch.cat([torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1), torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2)], -1)
        return (q_s, k_s, v_s)

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        if False:
            for i in range(10):
                print('nop')
        n_head = self.d_l
        (d_k, d_v) = (self.d_k, self.d_v)
        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        if self.q_as_matrix:
            q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        else:
            q_padded = q_s.repeat(mb_size, 1, 1)
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=DTYPE)
        for (i, (start, end)) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            if self.q_as_matrix:
                q_padded[:, i, :end - start, :] = q_s[:, start:end, :]
            k_padded[:, i, :end - start, :] = k_s[:, start:end, :]
            v_padded[:, i, :end - start, :] = v_s[:, start:end, :]
            invalid_mask[i, :end - start].fill_(False)
        if self.q_as_matrix:
            q_padded = q_padded.view(-1, len_padded, d_k)
            attn_mask = invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1)
        else:
            attn_mask = invalid_mask.unsqueeze(1).repeat(n_head, 1, 1)
        output_mask = (~invalid_mask).repeat(n_head, 1)
        return (q_padded, k_padded.view(-1, len_padded, d_k), v_padded.view(-1, len_padded, d_v), attn_mask, output_mask)

    def combine_v(self, outputs):
        if False:
            return 10
        d_l = self.d_l
        outputs = outputs.view(d_l, -1, self.d_v)
        if not self.partitioned:
            if self.combine_as_self:
                outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, d_l * self.d_v)
            else:
                outputs = torch.transpose(outputs, 0, 1)
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            if self.combine_as_self:
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, d_l * d_v1)
            else:
                outputs1 = torch.transpose(outputs1, 0, 1)
                outputs2 = torch.transpose(outputs2, 0, 1)
            outputs = torch.cat([self.proj1(outputs1), self.proj2(outputs2)], -1)
        return outputs

    def forward(self, inp, batch_idxs, k_inp=None):
        if False:
            while True:
                i = 10
        residual = inp
        len_inp = inp.size(0)
        (q_s, k_s, v_s) = self.split_qkv_packed(inp, k_inp=k_inp)
        (q_padded, k_padded, v_padded, attn_mask, output_mask) = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)
        (outputs_padded, attns_padded) = self.attention(q_padded, k_padded, v_padded, attn_mask=attn_mask)
        if not self.q_as_matrix:
            outputs_padded = outputs_padded.repeat(1, output_mask.size(-1), 1)
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)
        if self.use_resdrop:
            if self.combine_as_self:
                outputs = self.residual_dropout(outputs, batch_idxs)
            else:
                outputs = torch.cat([self.residual_dropout(outputs[:, i, :], batch_idxs).unsqueeze(1) for i in range(self.d_l)], 1)
        if self.combine_as_self:
            outputs = self.layer_norm(outputs + inp)
        else:
            for l in range(self.d_l):
                outputs[:, l, :] = outputs[:, l, :] + inp
            outputs = self.reduce_proj(outputs)
            outputs = self.layer_norm(outputs)
            outputs = outputs.view(len_inp, -1).contiguous()
        return (outputs, attns_padded)

class LabelAttentionModule(nn.Module):
    """
    Label Attention Module for label-specific representations
    The module can be used right after the Partitioned Attention, or it can be experimented with for the transition stack
    """

    def __init__(self, d_model, d_input_proj, d_k, d_v, d_l, d_proj, combine_as_self, use_resdrop=True, q_as_matrix=False, residual_dropout=0.1, attention_dropout=0.1, d_positional=None, d_ff=2048, relu_dropout=0.2, lattn_partitioned=True):
        if False:
            print('Hello World!')
        super().__init__()
        self.ff_dim = d_proj * d_l
        if not lattn_partitioned:
            self.d_positional = 0
        else:
            self.d_positional = d_positional if d_positional else 0
        if d_input_proj:
            if d_input_proj <= self.d_positional:
                raise ValueError('Illegal argument for d_input_proj: d_input_proj %d is smaller than d_positional %d' % (d_input_proj, self.d_positional))
            self.input_projection = nn.Linear(d_model - self.d_positional, d_input_proj - self.d_positional, bias=False)
            d_input = d_input_proj
        else:
            self.input_projection = None
            d_input = d_model
        self.label_attention = LabelAttention(d_input, d_k, d_v, d_l, d_proj, combine_as_self, use_resdrop, q_as_matrix, residual_dropout, attention_dropout, self.d_positional)
        if not lattn_partitioned:
            self.lal_ff = PositionwiseFeedForward(self.ff_dim, d_ff, relu_dropout, residual_dropout)
        else:
            self.lal_ff = PartitionedPositionwiseFeedForward(self.ff_dim, d_ff, self.d_positional, relu_dropout, residual_dropout)

    def forward(self, word_embeddings, tagged_word_lists):
        if False:
            i = 10
            return i + 15
        if self.input_projection:
            if self.d_positional > 0:
                word_embeddings = [torch.cat((self.input_projection(sentence[:, :-self.d_positional]), sentence[:, -self.d_positional:]), dim=1) for sentence in word_embeddings]
            else:
                word_embeddings = [self.input_projection(sentence) for sentence in word_embeddings]
        packed_len = sum((sentence.shape[0] for sentence in word_embeddings))
        batch_idxs = np.zeros(packed_len, dtype=int)
        batch_size = len(word_embeddings)
        i = 0
        sentence_lengths = [0] * batch_size
        for (sentence_idx, sentence) in enumerate(word_embeddings):
            sentence_lengths[sentence_idx] = len(sentence)
            for word in sentence:
                batch_idxs[i] = sentence_idx
                i += 1
        batch_indices = batch_idxs
        batch_idxs = BatchIndices(batch_idxs, word_embeddings[0].device)
        new_embeds = []
        for (sentence_idx, batch) in enumerate(word_embeddings):
            for (word_idx, embed) in enumerate(batch):
                if word_idx < sentence_lengths[sentence_idx]:
                    new_embeds.append(embed)
        new_word_embeddings = torch.stack(new_embeds)
        (labeled_representations, _) = self.label_attention(new_word_embeddings, batch_idxs)
        labeled_representations = self.lal_ff(labeled_representations, batch_idxs)
        final_labeled_representations = [[] for i in range(batch_size)]
        for (idx, embed) in enumerate(labeled_representations):
            final_labeled_representations[batch_indices[idx]].append(embed)
        for (idx, representation) in enumerate(final_labeled_representations):
            final_labeled_representations[idx] = torch.stack(representation)
        return final_labeled_representations