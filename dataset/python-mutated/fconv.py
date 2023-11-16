import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel, FairseqIncrementalDecoder, register_model, register_model_architecture
from fairseq.modules import AdaptiveSoftmax, BeamableMM, FairseqDropout, GradMultiply, LearnedPositionalEmbedding, LinearizedConvolution

@register_model('fconv')
class FConvModel(FairseqEncoderDecoderModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        if False:
            i = 10
            return i + 15

        def moses_subword(path):
            if False:
                i = 10
                return i + 15
            return {'path': path, 'tokenizer': 'moses', 'bpe': 'subword_nmt'}
        return {'conv.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2'), 'conv.wmt14.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2'), 'conv.wmt17.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2')}

    def __init__(self, encoder, decoder):
        if False:
            return 10
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum((layer is not None for layer in decoder.attention))

    @staticmethod
    def add_args(parser):
        if False:
            return 10
        'Add model-specific arguments to the parser.'
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR', help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR', help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR', help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true', help='share input and output embeddings (requires --decoder-out-embed-dim and --decoder-embed-dim to be equal)')

    @classmethod
    def build_model(cls, args, task):
        if False:
            return 10
        'Build a new model instance.'
        base_architecture(args)
        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)
        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)
        encoder = FConvEncoder(dictionary=task.source_dictionary, embed_dim=args.encoder_embed_dim, embed_dict=encoder_embed_dict, convolutions=eval(args.encoder_layers), dropout=args.dropout, max_positions=args.max_source_positions)
        decoder = FConvDecoder(dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim, embed_dict=decoder_embed_dict, convolutions=eval(args.decoder_layers), out_embed_dim=args.decoder_out_embed_dim, attention=eval(args.decoder_attention), dropout=args.dropout, max_positions=args.max_target_positions, share_embed=args.share_input_output_embed)
        return FConvModel(encoder, decoder)

class FConvEncoder(FairseqEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(self, dictionary, embed_dim=512, embed_dict=None, max_positions=1024, convolutions=((512, 3),) * 20, dropout=0.1):
        if False:
            print('Hello World!')
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.num_attention_layers = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, self.padding_idx)
        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []
        layer_in_channels = [in_channels]
        for (_, (out_channels, kernel_size, residual)) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels) if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(ConvTBC(in_channels, out_channels * 2, kernel_size, dropout=dropout, padding=padding))
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            src_tokens (LongTensor): tokens in the source language of shape\n                `(batch, src_len)`\n            src_lengths (LongTensor): lengths of each source sentence of shape\n                `(batch)`\n\n        Returns:\n            dict:\n                - **encoder_out** (tuple): a tuple with two elements, where the\n                  first element is the last encoder layer's output and the\n                  second element is the same quantity summed with the input\n                  embedding (used for attention). The shape of both tensors is\n                  `(batch, src_len, embed_dim)`.\n                - **encoder_padding_mask** (ByteTensor): the positions of\n                  padding elements of shape `(batch, src_len)`\n        "
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x
        x = self.fc1(x)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        x = x.transpose(0, 1)
        residuals = [x]
        for (proj, conv, res_layer) in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            x = self.dropout_module(x)
            if conv.kernel_size[0] % 2 == 1:
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)
        x = x.transpose(1, 0)
        x = self.fc2(x)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))
        y = (x + input_embedding) * math.sqrt(0.5)
        return {'encoder_out': (x, y), 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        if False:
            for i in range(10):
                print('nop')
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = (encoder_out['encoder_out'][0].index_select(0, new_order), encoder_out['encoder_out'][1].index_select(0, new_order))
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        if False:
            while True:
                i = 10
        'Maximum input length supported by the encoder.'
        return self.embed_positions.max_positions

class AttentionLayer(nn.Module):

    def __init__(self, conv_channels, embed_dim, bmm=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        if False:
            while True:
                i = 10
        residual = x
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(x)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x
        x = self.bmm(x, encoder_out[1])
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(dim=1, keepdim=True)
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return (x, attn_scores)

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        if False:
            print('Hello World!')
        'Replace torch.bmm with BeamableMM.'
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))

class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""

    def __init__(self, dictionary, embed_dim=512, embed_dict=None, out_embed_dim=256, max_positions=1024, convolutions=((512, 3),) * 20, attention=True, dropout=0.1, share_embed=False, positional_embeddings=True, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0.0):
        if False:
            i = 10
            return i + 15
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.need_attn = True
        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of length equal to the number of layers.')
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx) if positional_embeddings else None
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []
        layer_in_channels = [in_channels]
        for (i, (out_channels, kernel_size, residual)) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels) if residual_dim != out_channels else None)
            self.convolutions.append(LinearizedConv1d(in_channels, out_channels * 2, kernel_size, padding=kernel_size - 1, dropout=dropout))
            self.attention.append(AttentionLayer(out_channels, embed_dim) if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None
        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, in_channels, adaptive_softmax_cutoff, dropout=adaptive_softmax_dropout)
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, 'Shared embed weights implies same dimensions  out_embed_dim={} vs embed_dim={}'.format(out_embed_dim, embed_dim)
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        if False:
            print('Hello World!')
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
            (encoder_a, encoder_b) = self._split_encoder_out(encoder_out, incremental_state)
        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)
        x += pos_embed
        x = self.dropout_module(x)
        target_embedding = x
        x = self.fc1(x)
        x = self._transpose_if_training(x, incremental_state)
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for (proj, conv, attention, res_layer) in zip(self.projections, self.convolutions, self.attention, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None
            x = self.dropout_module(x)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)
                (x, attn_scores) = attention(x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)
                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)
                x = self._transpose_if_training(x, incremental_state)
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)
        x = self._transpose_if_training(x, incremental_state)
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.fc3(x)
        return (x, avg_attn_scores)

    def reorder_incremental_state(self, incremental_state, new_order):
        if False:
            i = 10
            return i + 15
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if encoder_out is not None:
            encoder_out = tuple((eo.index_select(0, new_order) for eo in encoder_out))
            utils.set_incremental_state(self, incremental_state, 'encoder_out', encoder_out)

    def max_positions(self):
        if False:
            return 10
        'Maximum output length supported by the decoder.'
        return self.embed_positions.max_positions if self.embed_positions is not None else float('inf')

    def upgrade_state_dict(self, state_dict):
        if False:
            while True:
                i = 10
        if utils.item(state_dict.get('decoder.version', torch.Tensor([1]))[0]) < 2:
            for (i, conv) in enumerate(self.convolutions):
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def make_generation_fast_(self, need_attn=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if False:
            print('Hello World!')
        if incremental_state is not None:
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        if False:
            while True:
                i = 10
        'Split and transpose encoder outputs.\n\n        This is cached when doing incremental inference.\n        '
        cached_result = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result
        (encoder_a, encoder_b) = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)
        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if False:
            while True:
                i = 10
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

def extend_conv_spec(convolutions):
    if False:
        while True:
            i = 10
    '\n    Extends convolutional spec that is a list of tuples of 2 or 3 parameters\n    (kernel size, dim size and optionally how many layers behind to look for residual)\n    to default the residual propagation param if it is not specified\n    '
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    if False:
        print('Hello World!')
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    if False:
        print('Hello World!')
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, dropout=0.0):
    if False:
        print('Hello World!')
    'Weight-normalized Linear layer (input: N x T x C)'
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)

def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    if False:
        i = 10
        return i + 15
    'Weight-normalized Conv1d layer optimized for decoding'
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(4 * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

def ConvTBC(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    if False:
        i = 10
        return i + 15
    'Weight-normalized Conv1d layer'
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(4 * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

@register_model_architecture('fconv', 'fconv')
def base_architecture(args):
    if False:
        for i in range(10):
            print('nop')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)

@register_model_architecture('fconv', 'fconv_iwslt_de_en')
def fconv_iwslt_de_en(args):
    if False:
        print('Hello World!')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3)] * 4')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(256, 3)] * 3')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    base_architecture(args)

@register_model_architecture('fconv', 'fconv_wmt_en_ro')
def fconv_wmt_en_ro(args):
    if False:
        return 10
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    base_architecture(args)

@register_model_architecture('fconv', 'fconv_wmt_en_de')
def fconv_wmt_en_de(args):
    if False:
        for i in range(10):
            print('nop')
    convs = '[(512, 3)] * 9'
    convs += ' + [(1024, 3)] * 4'
    convs += ' + [(2048, 1)] * 2'
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_layers = getattr(args, 'encoder_layers', convs)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_layers = getattr(args, 'decoder_layers', convs)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    base_architecture(args)

@register_model_architecture('fconv', 'fconv_wmt_en_fr')
def fconv_wmt_en_fr(args):
    if False:
        for i in range(10):
            print('nop')
    convs = '[(512, 3)] * 6'
    convs += ' + [(768, 3)] * 4'
    convs += ' + [(1024, 3)] * 3'
    convs += ' + [(2048, 1)] * 1'
    convs += ' + [(4096, 1)] * 1'
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_layers = getattr(args, 'encoder_layers', convs)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_layers = getattr(args, 'decoder_layers', convs)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    base_architecture(args)