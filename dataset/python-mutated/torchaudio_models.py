import math
from collections import OrderedDict
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
__all__ = ['Wav2Letter']

class Wav2Letter(nn.Module):
    """Wav2Letter model architecture from the `"Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
     <https://arxiv.org/abs/1609.03193>`_ paper.
     :math:`\\text{padding} = \\frac{\\text{ceil}(\\text{kernel} - \\text{stride})}{2}`
    Args:
        num_classes (int, optional): Number of classes to be classified. (Default: ``40``)
        input_type (str, optional): Wav2Letter can use as input: ``waveform``, ``power_spectrum``
         or ``mfcc`` (Default: ``waveform``).
        num_features (int, optional): Number of input features that the network will receive (Default: ``1``).
    """

    def __init__(self, num_classes: int=40, input_type: str='waveform', num_features: int=1) -> None:
        if False:
            return 10
        super().__init__()
        acoustic_num_features = 250 if input_type == 'waveform' else num_features
        acoustic_model = nn.Sequential(nn.Conv1d(in_channels=acoustic_num_features, out_channels=250, kernel_size=48, stride=2, padding=23), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=True), nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16), nn.ReLU(inplace=True), nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        if input_type == 'waveform':
            waveform_model = nn.Sequential(nn.Conv1d(in_channels=num_features, out_channels=250, kernel_size=250, stride=160, padding=45), nn.ReLU(inplace=True))
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)
        if input_type in ['power_spectrum', 'mfcc']:
            self.acoustic_model = acoustic_model

    def forward(self, x: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            x (Tensor): Tensor of dimension (batch_size, num_features, input_length).\n        Returns:\n            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).\n        '
        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class SequenceWise(nn.Module):

    def __init__(self, module):
        if False:
            i = 10
            return i + 15
        '\n        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.\n        Allows handling of variable sequence lengths and minibatch sizes.\n        :param module: Module to apply input to.\n        '
        super().__init__()
        self.module = module

    def forward(self, x):
        if False:
            return 10
        (t, n) = (x.size(0), x.size(1))
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class MaskConv(nn.Module):

    def __init__(self, seq_module):
        if False:
            return 10
        '\n        Adds padding to the output of the module based on the given lengths. This is to ensure that the\n        results of the model do not change when batch sizes change during inference.\n        Input needs to be in the shape of (BxCxDxT)\n        :param seq_module: The sequential module containing the conv stack.\n        '
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        if False:
            print('Hello World!')
        '\n        :param x: The input of size BxCxDxT\n        :param lengths: The actual length of each sequence in the batch\n        :return: Masked output from the module\n        '
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for (i, length) in enumerate(lengths):
                length = length.item()
                if mask[i].size(2) - length > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return (x, lengths)

class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if False:
            for i in range(10):
                print('nop')
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        if False:
            print('Hello World!')
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        if False:
            i = 10
            return i + 15
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if False:
            print('Hello World!')
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        (x, h) = self.rnn(x)
        (x, _) = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x

class Lookahead(nn.Module):

    def __init__(self, n_features, context):
        if False:
            while True:
                i = 10
        super().__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1, groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        if False:
            return 10
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__ + '(' + 'n_features=' + str(self.n_features) + ', context=' + str(self.context) + ')'

class DeepSpeech(nn.Module):

    def __init__(self, rnn_type, labels, rnn_hidden_size, nb_layers, audio_conf, bidirectional, context=20):
        if False:
            return 10
        super().__init__()
        self.hidden_size = rnn_hidden_size
        self.hidden_layers = nb_layers
        self.rnn_type = rnn_type
        self.audio_conf = audio_conf
        self.labels = labels
        self.bidirectional = bidirectional
        sample_rate = self.audio_conf['sample_rate']
        window_size = self.audio_conf['window_size']
        num_classes = len(self.labels)
        self.conv = MaskConv(nn.Sequential(nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)), nn.BatchNorm2d(32), nn.Hardtanh(0, 20, inplace=True), nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)), nn.BatchNorm2d(32), nn.Hardtanh(0, 20, inplace=True)))
        rnn_input_size = int(math.floor(sample_rate * window_size / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(Lookahead(rnn_hidden_size, context=context), nn.Hardtanh(0, 20, inplace=True)) if not bidirectional else None
        fully_connected = nn.Sequential(nn.BatchNorm1d(rnn_hidden_size), nn.Linear(rnn_hidden_size, num_classes, bias=False))
        self.fc = nn.Sequential(SequenceWise(fully_connected))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        if False:
            for i in range(10):
                print('nop')
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        (x, _) = self.conv(x, output_lengths)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
        if not self.bidirectional:
            x = self.lookahead(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return (x, output_lengths)

    def get_seq_lens(self, input_length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable\n        containing the size sequences that will be output by the network.\n        :param input_length: 1D Tensor\n        :return: 1D Tensor scaled by model\n        '
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                seq_len = seq_len.true_divide(m.stride[1]) + 1
        return seq_len.int()

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if False:
            return 10
        'Inputs of forward function\n        Args:\n            x: the sequence fed to the positional encoder model (required).\n        Shape:\n            x: [sequence length, batch size, embed dim]\n            output: [sequence length, batch size, embed dim]\n        Examples:\n            >>> output = pos_encoder(x)\n        '
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        if False:
            while True:
                i = 10
        super().__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except Exception as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        if False:
            for i in range(10):
                print('nop')
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if False:
            while True:
                i = 10
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

class MultiheadAttentionContainer(torch.nn.Module):

    def __init__(self, nhead, in_proj_container, attention_layer, out_proj):
        if False:
            return 10
        'A multi-head attention container\n        Args:\n            nhead: the number of heads in the multiheadattention model\n            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).\n            attention_layer: The attention layer.\n            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).\n        Examples::\n            >>> import torch\n            >>> embed_dim, num_heads, bsz = 10, 5, 64\n            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),\n                                                    torch.nn.Linear(embed_dim, embed_dim),\n                                                    torch.nn.Linear(embed_dim, embed_dim))\n            >>> MHA = MultiheadAttentionContainer(num_heads,\n                                                  in_proj_container,\n                                                  ScaledDotProduct(),\n                                                  torch.nn.Linear(embed_dim, embed_dim))\n            >>> query = torch.rand((21, bsz, embed_dim))\n            >>> key = value = torch.rand((16, bsz, embed_dim))\n            >>> attn_output, attn_weights = MHA(query, key, value)\n            >>> print(attn_output.shape)\n            >>> torch.Size([21, 64, 10])\n        '
        super().__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, bias_k: Optional[torch.Tensor]=None, bias_v: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            query, key, value (Tensor): map a query and a set of key-value pairs to an output.\n                See "Attention Is All You Need" for more details.\n            attn_mask, bias_k and bias_v (Tensor, optional): keyword arguments passed to the attention layer.\n                See the definitions in the attention.\n        Shape:\n            - Inputs:\n            - query: :math:`(L, N, E)`\n            - key: :math:`(S, N, E)`\n            - value: :math:`(S, N, E)`\n            - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.\n            - Outputs:\n            - attn_output: :math:`(L, N, E)`\n            - attn_output_weights: :math:`(N * H, L, S)`\n            where where L is the target length, S is the sequence length, H is the number of attention heads,\n                N is the batch size, and E is the embedding dimension.\n        '
        (tgt_len, src_len, bsz, embed_dim) = (query.size(-3), key.size(-3), query.size(-2), query.size(-1))
        (q, k, v) = self.in_proj_container(query, key, value)
        assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
        head_dim = q.size(-1) // self.nhead
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)
        assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
        head_dim = k.size(-1) // self.nhead
        k = k.reshape(src_len, bsz * self.nhead, head_dim)
        assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
        head_dim = v.size(-1) // self.nhead
        v = v.reshape(src_len, bsz * self.nhead, head_dim)
        (attn_output, attn_output_weights) = self.attention_layer(q, k, v, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_output_weights)

class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0):
        if False:
            print('Hello World!')
        'Processes a projected query and key-value pair to apply\n        scaled dot product attention.\n        Args:\n            dropout (float): probability of dropping an attention weight.\n        Examples::\n            >>> SDP = torchtext.models.ScaledDotProduct(0.1)\n            >>> q = torch.randn(256, 21, 3)\n            >>> k = v = torch.randn(256, 21, 3)\n            >>> attn_output, attn_weights = SDP(q, k, v)\n            >>> print(attn_output.shape, attn_weights.shape)\n            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])\n        '
        super().__init__()
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, bias_k: Optional[torch.Tensor]=None, bias_v: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        'Uses a scaled dot product with the projected key-value pair to update\n        the projected query.\n        Args:\n            query (Tensor): Projected query\n            key (Tensor): Projected key\n            value (Tensor): Projected value\n            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.\n            bias_k and bias_v: (Tensor, optional): one more key and value sequence to be added at\n                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide\n                non-None to both arguments in order to activate them.\n        Shape:\n            - query: :math:`(L, N * H, E / H)`\n            - key: :math:`(S, N * H, E / H)`\n            - value: :math:`(S, N * H, E / H)`\n            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend\n                while ``False`` values will be unchanged.\n            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`\n            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`\n            where L is the target length, S is the source length, H is the number\n            of attention heads, N is the batch size, and E is the embedding dimension.\n        '
        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and (bias_k.size(-3) == 1), 'Shape of bias_k is not supported'
            assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and (bias_v.size(-3) == 1), 'Shape of bias_v is not supported'
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                _attn_mask = attn_mask
                attn_mask = torch.nn.functional.pad(_attn_mask, [0, 1])
        (tgt_len, head_dim) = (query.size(-3), query.size(-1))
        assert query.size(-1) == key.size(-1) == value.size(-1), 'The feature dim of query, key, value must be equal.'
        assert key.size() == value.size(), 'Shape of key, value must match'
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))
        (query, key, value) = (query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3))
        query = query * float(head_dim) ** (-0.5)
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if attn_mask.size(-1) != src_len or attn_mask.size(-2) != tgt_len or (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads):
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError('Only bool tensor is supported for attn_mask')
        attn_output_weights = torch.matmul(query, key.mT)
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -100000000.0)
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return (attn_output.transpose(-2, -3), attn_output_weights)

class InProjContainer(torch.nn.Module):

    def __init__(self, query_proj, key_proj, value_proj):
        if False:
            return 10
        'A in-proj container to process inputs.\n        Args:\n            query_proj: a proj layer for query.\n            key_proj: a proj layer for key.\n            value_proj: a proj layer for value.\n        '
        super().__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            return 10
        'Projects the input sequences using in-proj layers.\n        Args:\n            query, key, value (Tensors): sequence to be projected\n        Shape:\n            - query, key, value: :math:`(S, N, E)`\n            - Output: :math:`(S, N, E)`\n            where S is the sequence length, N is the batch size, and E is the embedding dimension.\n        '
        return (self.query_proj(query), self.key_proj(key), self.value_proj(value))