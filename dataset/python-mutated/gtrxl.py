from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
from ding.torch_utils.network.nn_module import fc_block, build_normalization, F

class PositionalEmbedding(nn.Module):
    """
    Overview:
        Positional Embedding used in vanilla Transformer
    .. note::
        Adapted from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
    """

    def __init__(self, embedding_dim: int):
        if False:
            i = 10
            return i + 15
        '\n        Arguments:\n            - embedding_dim: (:obj:`int`): dimension of embedding\n        '
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1 / 10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Compute positional embedding\n        Arguments:\n            - pos_seq: (:obj:`torch.Tensor`): positional sequence,\n             usually a 1D integer sequence as [seq_len-1, seq_len-2, ..., 1, 0],\n        Returns:\n            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)\n        '
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_embedding.unsqueeze(1)

class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
    """

    def __init__(self, input_dim: int, bg: float=2.0):
        if False:
            i = 10
            return i + 15
        '\n        Arguments:\n            - input_dim: (:obj:`int`): dimension of input.\n            - bg (:obj:`bg`): gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to\n            be close to the identity map. This can greatly improve the learning speed and stability since it\n            initializes the agent close to a Markovian policy (ignore attention at the beginning).\n        '
        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Compute output value with gating mechanism\n        Arguments:\n            - x: (:obj:`torch.Tensor`): first input.\n            - y: (:obj:`torch.Tensor`): second input.\n            x and y have same shape and last shape is input_dim.\n        Returns:\n            - g: (:obj:`torch.Tensor`): output of GRU. Same shape of x and y.\n        '
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g

class Memory:
    """
    Overview:
        Stores the context used to add memory to Transformer.
    .. note::
        For details refer to Transformer-XL: https://arxiv.org/abs/1901.02860
    """

    def __init__(self, memory_len: int=20, batch_size: int=64, embedding_dim: int=256, layer_num: int=3, memory: Optional[torch.Tensor]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Arguments:\n            - memory_len (:obj:`int`): dimension of memory (how many past observations to use as memory)\n            - batch_size (:obj:`int`): dimension of each batch\n            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)\n            - layer_num (:obj:`int`): number of transformer layers\n        '
        super(Memory, self).__init__()
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.layer_num = layer_num
        self.memory_len = memory_len
        self.memory = None
        self.init(memory)

    def init(self, memory: Optional[torch.Tensor]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Init memory with an input list of tensors or create it automatically given its dimensions.\n        Arguments:\n            - memory: (:obj:`Optional[torch.Tensor]`): memory input.\n            Shape is (layer_num, memory_len, bs, embedding_dim).\n            memory_len is length of memory, bs is batch size and embedding_dim is the dimension of embedding.\n        '
        if memory is not None:
            self.memory = memory
            (layer_num_plus1, self.memory_len, self.bs, self.embedding_dim) = memory.shape
            self.layer_num = layer_num_plus1 - 1
        else:
            self.memory = torch.zeros(self.layer_num + 1, self.memory_len, self.bs, self.embedding_dim, dtype=torch.float)

    def update(self, hidden_state: List[torch.Tensor]):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Update the memory given a sequence of hidden states.\n        Example for single layer:\n\n            memory_len=3, hidden_size_len=2, bs=3\n\n                m00 m01 m02      h00 h01 h02              m20 m21 m22\n            m = m10 m11 m12  h = h10 h11 h12  => new_m =  h00 h01 h02\n                m20 m21 m22                               h10 h11 h12\n        Arguments:\n            - hidden_state: (:obj:`List[torch.Tensor]`): hidden states to update the memory.\n            Shape is (cur_seq, bs, embedding_dim) for each layer. cur_seq is length of sequence.\n        Returns:\n            - memory: (:obj:`Optional[torch.Tensor]`): output memory.\n            Shape is (layer_num, memory_len, bs, embedding_dim).\n        '
        if self.memory is None or hidden_state is None:
            raise ValueError('Failed to update memory! Memory would be None')
        sequence_len = hidden_state[0].shape[0]
        with torch.no_grad():
            new_memory = []
            end = self.memory_len + sequence_len
            beg = max(0, end - self.memory_len)
            for i in range(self.layer_num + 1):
                m = self.memory[i]
                h = hidden_state[i]
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg:end].detach())
        new_memory = torch.stack(new_memory, dim=0)
        self.memory = new_memory
        return new_memory

    def get(self):
        if False:
            return 10
        '\n        Overview:\n            Memory getter method.\n        Returns:\n            - memory: (:obj:`Optional[torch.Tensor]`): output memory.\n            Shape is (layer_num, memory_len, bs, embedding_dim).\n        '
        return self.memory

    def to(self, device: str='cpu'):
        if False:
            for i in range(10):
                print('nop')
        self.memory = self.memory.to(device)

class AttentionXL(torch.nn.Module):
    """
    Overview:
        Attention of TransformerXL.
    """

    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Overview:\n            Init AttentionXL.\n        Arguments:\n            - input_dim (:obj:`int`): dimension of input\n            - head_dim (:obj:`int`): dimension of each head\n            - head_num (:obj:`int`): number of heads for multihead attention\n            - dropout (:obj:`nn.Module`): dropout function\n        '
        super(AttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = fc_block(input_dim, head_dim * head_num * 2)
        self.attention_q = fc_block(input_dim, head_dim * head_num)
        self.project = fc_block(head_dim * head_num, input_dim)
        self.project_pos = fc_block(input_dim, head_dim * head_num)
        self.scale = 1 / head_dim ** 0.5

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool=False):
        if False:
            return 10
        '\n        Overview:\n            Relatively shift the attention score matrix.\n        Example:\n            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0\n            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0\n            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22\n                                                a20 a21 a22\n            1) Append one "column" of zeros to the left\n            2) Reshape the matrix from [3 x 4] into [4 x 3]\n            3) Remove the first "row"\n            4) Mask out the upper triangle (optional)\n        .. note::\n            See the following material for better understanding:\n                https://github.com/kimiyoung/transformer-xl/issues/8\n                https://arxiv.org/pdf/1901.02860.pdf (Appendix B)\n        Arguments:\n            - x (:obj:`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).\n            - zero_upper (:obj:`bool`): if True set the upper-right triangle to zero.\n        Returns:\n            - x (:obj:`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).\n        '
        x_padded = F.pad(x, [1, 0])
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))
        return x

    def forward(self, inputs: torch.Tensor, pos_embedding: torch.Tensor, full_input: torch.Tensor, u: torch.nn.Parameter, v: torch.nn.Parameter, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            return 10
        'Overview:\n            Compute AttentionXL.\n        Arguments:\n            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)\n            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)\n            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)\n            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)\n            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)\n            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)\n            full_seq = prev_seq + cur_seq\n        Returns:\n            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)\n        '
        (bs, cur_seq, full_seq) = (inputs.shape[1], inputs.shape[0], full_input.shape[0])
        prev_seq = full_seq - cur_seq
        kv = self.attention_kv(full_input)
        (key, value) = torch.chunk(kv, 2, dim=-1)
        query = self.attention_q(inputs)
        r = self.project_pos(pos_embedding)
        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)
        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)
        position_attn = self._rel_shift(position_attn)
        attn = content_attn + position_attn
        attn.mul_(self.scale)
        if mask is not None and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)
            assert mask.shape[2:] == attn.shape[2:]
            attn = attn.masked_fill(mask, -float('inf')).type_as(attn)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)
        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        output = self.dropout(self.project(attn_vec))
        return output

class GatedTransformerXLLayer(torch.nn.Module):
    """
    Overview:
        Attention layer of GTrXL
    """

    def __init__(self, input_dim: int, head_dim: int, hidden_dim: int, head_num: int, mlp_num: int, dropout: nn.Module, activation: nn.Module, gru_gating: bool=True, gru_bias: float=2.0) -> None:
        if False:
            print('Hello World!')
        '\n        Arguments:\n            - input_dim (:obj:`int`): dimension of input\n            - head_dim (:obj:`int`): dimension of each head\n            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp\n            - head_num (:obj:`int`): number of heads for multihead attention\n            - mlp_num (:obj:`int`): number of mlp layers in attention layer\n            - dropout (:obj:`nn.Module`): dropout\n            - activation (:obj:`nn.Module`): activation function\n            - gru_gating (:obj:`bool`): if False replace GRU gates with residual connections\n            - gru_bias (:obj:`float`): GRU gate bias\n        '
        super(GatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
        self.attention = AttentionXL(input_dim, head_dim, head_num, dropout)
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.layernorm2 = build_normalization('LN')(input_dim)
        self.activation = activation

    def forward(self, inputs: torch.Tensor, pos_embedding: torch.Tensor, u: torch.nn.Parameter, v: torch.nn.Parameter, memory: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Overview:\n            Compute forward pass of GTrXL layer.\n        Arguments:\n            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)\n            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)\n            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)\n            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)\n            - memory (:obj:`Optional[torch.Tensor]`): memory of shape (prev_seq, bs, input_dim)\n            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)\n            full_seq = prev_seq + cur_seq\n        Returns:\n            - output (:obj:`torch.Tensor`): layer output of shape (cur_seq, bs, input_dim)\n        '
        full_input = torch.cat([memory, inputs], dim=0)
        x1 = self.layernorm1(full_input)
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2

class GTrXL(nn.Module):
    """
    Overview:
        GTrXL Transformer implementation.

    .. note::
        For details refer to Stabilizing Transformer for Reinforcement Learning: https://arxiv.org/abs/1910.06764
    """

    def __init__(self, input_dim: int, head_dim: int=128, embedding_dim: int=256, head_num: int=2, mlp_num: int=2, layer_num: int=3, memory_len: int=64, dropout_ratio: float=0.0, activation: nn.Module=nn.ReLU(), gru_gating: bool=True, gru_bias: float=2.0, use_embedding_layer: bool=True) -> None:
        if False:
            return 10
        "Overview:\n            Init GTrXL Model\n        Arguments:\n            - input_dim (:obj:`int`): dimension of input (dimension of a single observation)\n            - head_dim (:obj:`int`): dimension of each head\n            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp\n            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)\n            - head_num (:obj:`int`): number of heads for multihead attention\n            - mlp_num (:obj:`int`): number of mlp layers in attention layer\n            - layer_num (:obj:`int`): number of transformer layers\n            - dropout_ratio (:obj:`float`): dropout ratio\n            - activation (:obj:`nn.Module`): activation function\n            - gru_gating (:obj:`bool`): if False replace GRU gates with residual connections\n            - gru_bias (:obj:`float`): GRU gate bias\n            - use_embedding_layer (:obj:`bool`): default True. If False, don't use input embedding layer.\n        "
        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        self.activation = activation
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        self.memory = None
        self.memory_len = memory_len
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(GatedTransformerXLLayer(dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, self.activation, gru_gating, gru_bias))
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        (self.u, self.v) = (torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)), torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)))
        self.att_mask = {}
        self.pos_embedding_dict = {}

    def reset_memory(self, batch_size: Optional[int]=None, state: Optional[torch.Tensor]=None):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Clear or set the memory of GTrXL.\n        Arguments:\n            - batch_size (:obj:`Optional[int]`): batch size\n            - state (:obj:`Optional[torch.Tensor]`): input memory. Shape is (layer_num, memory_len, bs, embedding_dim).\n        '
        self.memory = Memory(memory_len=self.memory_len, layer_num=self.layer_num, embedding_dim=self.embedding_dim)
        if batch_size is not None:
            self.memory = Memory(self.memory_len, batch_size, self.embedding_dim, self.layer_num)
        elif state is not None:
            self.memory.init(state)

    def get_memory(self):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Returns memory of GTrXL.\n        Returns:\n            - memory: (:obj:`Optional[torch.Tensor]`): output memory or None if memory has not been initialized.                 Shape is (layer_num, memory_len, bs, embedding_dim).\n        '
        if self.memory is None:
            return None
        else:
            return self.memory.get()

    def forward(self, x: torch.Tensor, batch_first: bool=False, return_mem: bool=True) -> Dict[str, torch.Tensor]:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            GTrXL forward pass.\n        Arguments:\n            - x (:obj:`torch.Tensor`): input tensor. Shape (seq_len, bs, input_size).\n            - batch_first (:obj:`bool`): if the input data has shape (bs, seq_len, input_size), set this param to                 ``True`` in order to transpose along the first and second dimension and obtain shape                 (seq_len, bs, input_size). This param doesn't affects the output memory.\n            - return_mem (:obj:`bool`): if this param is False, return only the output tensor without dict.\n        Returns:\n            - x (:obj:`Dict[str, torch.Tensor]`): dict containing transformer output of shape              (seq_len, bs, embedding_size) and memory of shape (layer_num, seq_len, bs, embedding_size)\n        "
        if batch_first:
            x = torch.transpose(x, 1, 0)
        (cur_seq, bs) = x.shape[:2]
        memory = None if self.memory is None else self.memory.get()
        if memory is None:
            self.reset_memory(bs)
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
            warnings.warn("Memory {} and Input {} dimensions don't match, this will cause the memory to be initialized to fit your input!".format(list(memory.shape[-2:]), [x.shape[-2]] + [self.embedding_dim]))
            self.reset_memory(bs)
        self.memory.to(x.device)
        memory = self.memory.get()
        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq
        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = torch.triu(torch.ones((cur_seq, full_seq)), diagonal=1 + prev_seq).bool().unsqueeze(-1).to(x.device)
            self.att_mask[cur_seq] = attn_mask
        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)
        hidden_state = [x]
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out, pos_embedding, self.u, self.v, mask=attn_mask, memory=memory[i])
            hidden_state.append(out.clone())
        out = self.dropout(out)
        self.memory.update(hidden_state)
        if batch_first:
            out = torch.transpose(out, 1, 0)
        if return_mem:
            output = {'logit': out, 'memory': memory}
        else:
            output = {'logit': out}
        return output