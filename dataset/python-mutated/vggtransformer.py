import argparse
import math
from collections.abc import Iterable
import torch
import torch.nn as nn
from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel, FairseqEncoderModel, FairseqIncrementalDecoder, register_model, register_model_architecture
from fairseq.modules import LinearizedConvolution, TransformerDecoderLayer, TransformerEncoderLayer, VGGBlock

@register_model('asr_vggtransformer')
class VGGTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """

    def __init__(self, encoder, decoder):
        if False:
            print('Hello World!')
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        'Add model-specific arguments to the parser.'
        parser.add_argument('--input-feat-per-channel', type=int, metavar='N', help='encoder input dimension per input channel')
        parser.add_argument('--vggblock-enc-config', type=str, metavar='EXPR', help='\n    an array of tuples each containing the configuration of one vggblock:\n    [(out_channels,\n      conv_kernel_size,\n      pooling_kernel_size,\n      num_conv_layers,\n      use_layer_norm), ...])\n            ')
        parser.add_argument('--transformer-enc-config', type=str, metavar='EXPR', help='"\n    a tuple containing the configuration of the encoder transformer layers\n    configurations:\n    [(input_dim,\n      num_heads,\n      ffn_dim,\n      normalize_before,\n      dropout,\n      attention_dropout,\n      relu_dropout), ...]\')\n            ')
        parser.add_argument('--enc-output-dim', type=int, metavar='N', help='\n    encoder output dimension, can be None. If specified, projecting the\n    transformer output to the specified dimension')
        parser.add_argument('--in-channels', type=int, metavar='N', help='number of encoder input channels')
        parser.add_argument('--tgt-embed-dim', type=int, metavar='N', help='embedding dimension of the decoder target tokens')
        parser.add_argument('--transformer-dec-config', type=str, metavar='EXPR', help='\n    a tuple containing the configuration of the decoder transformer layers\n    configurations:\n    [(input_dim,\n      num_heads,\n      ffn_dim,\n      normalize_before,\n      dropout,\n      attention_dropout,\n      relu_dropout), ...]\n            ')
        parser.add_argument('--conv-dec-config', type=str, metavar='EXPR', help='\n    an array of tuples for the decoder 1-D convolution config\n        [(out_channels, conv_kernel_size, use_layer_norm), ...]')

    @classmethod
    def build_encoder(cls, args, task):
        if False:
            while True:
                i = 10
        return VGGTransformerEncoder(input_feat_per_channel=args.input_feat_per_channel, vggblock_config=eval(args.vggblock_enc_config), transformer_config=eval(args.transformer_enc_config), encoder_output_dim=args.enc_output_dim, in_channels=args.in_channels)

    @classmethod
    def build_decoder(cls, args, task):
        if False:
            while True:
                i = 10
        return TransformerDecoder(dictionary=task.target_dictionary, embed_dim=args.tgt_embed_dim, transformer_config=eval(args.transformer_dec_config), conv_config=eval(args.conv_dec_config), encoder_output_dim=args.enc_output_dim)

    @classmethod
    def build_model(cls, args, task):
        if False:
            print('Hello World!')
        'Build a new model instance.'
        base_architecture(args)
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            while True:
                i = 10
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs
DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2

def prepare_transformer_encoder_params(input_dim, num_heads, ffn_dim, normalize_before, dropout, attention_dropout, relu_dropout):
    if False:
        print('Hello World!')
    args = argparse.Namespace()
    args.encoder_embed_dim = input_dim
    args.encoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.encoder_normalize_before = normalize_before
    args.encoder_ffn_embed_dim = ffn_dim
    return args

def prepare_transformer_decoder_params(input_dim, num_heads, ffn_dim, normalize_before, dropout, attention_dropout, relu_dropout):
    if False:
        return 10
    args = argparse.Namespace()
    args.encoder_embed_dim = None
    args.decoder_embed_dim = input_dim
    args.decoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.decoder_normalize_before = normalize_before
    args.decoder_ffn_embed_dim = ffn_dim
    return args

class VGGTransformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(self, input_feat_per_channel, vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG, transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG, encoder_output_dim=512, in_channels=1, transformer_context=None, transformer_sampling=None):
        if False:
            for i in range(10):
                print('nop')
        'constructor for VGGTransformerEncoder\n\n        Args:\n            - input_feat_per_channel: feature dim (not including stacked,\n              just base feature)\n            - in_channel: # input channels (e.g., if stack 8 feature vector\n                together, this is 8)\n            - vggblock_config: configuration of vggblock, see comments on\n                DEFAULT_ENC_VGGBLOCK_CONFIG\n            - transformer_config: configuration of transformer layer, see comments\n                on DEFAULT_ENC_TRANSFORMER_CONFIG\n            - encoder_output_dim: final transformer output embedding dimension\n            - transformer_context: (left, right) if set, self-attention will be focused\n              on (t-left, t+right)\n            - transformer_sampling: an iterable of int, must match with\n              len(transformer_config), transformer_sampling[i] indicates sampling\n              factor for i-th transformer layer, after multihead att and feedfoward\n              part\n        '
        super().__init__(None)
        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError('vggblock_config is not iterable')
            self.num_vggblocks = len(vggblock_config)
        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.pooling_kernel_sizes = []
        if vggblock_config is not None:
            for (_, config) in enumerate(vggblock_config):
                (out_channels, conv_kernel_size, pooling_kernel_size, num_conv_layers, layer_norm) = config
                self.conv_layers.append(VGGBlock(in_channels, out_channels, conv_kernel_size, pooling_kernel_size, num_conv_layers, input_dim=input_feat_per_channel, layer_norm=layer_norm))
                self.pooling_kernel_sizes.append(pooling_kernel_size)
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim
        transformer_input_dim = self.infer_conv_output_dim(self.in_channels, self.input_dim)
        self.validate_transformer_config(transformer_config)
        self.transformer_context = self.parse_transformer_context(transformer_context)
        self.transformer_sampling = self.parse_transformer_sampling(transformer_sampling, len(transformer_config))
        self.transformer_layers = nn.ModuleList()
        if transformer_input_dim != transformer_config[0][0]:
            self.transformer_layers.append(Linear(transformer_input_dim, transformer_config[0][0]))
        self.transformer_layers.append(TransformerEncoderLayer(prepare_transformer_encoder_params(*transformer_config[0])))
        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(Linear(transformer_config[i - 1][0], transformer_config[i][0]))
            self.transformer_layers.append(TransformerEncoderLayer(prepare_transformer_encoder_params(*transformer_config[i])))
        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers.extend([Linear(transformer_config[-1][0], encoder_output_dim), LayerNorm(encoder_output_dim)])

    def forward(self, src_tokens, src_lengths, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        src_tokens: padded tensor (B, T, C * feat)\n        src_lengths: tensor of original lengths of input utterances (B,)\n        '
        (bsz, max_seq_len, _) = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)
        (bsz, _, output_seq_len, _) = x.size()
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)
        input_lengths = src_lengths.clone()
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float() / s).ceil().long()
        (encoder_padding_mask, _) = lengths_to_encoder_padding_mask(input_lengths, batch_first=True)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)
        transformer_layer_idx = 0
        for layer_idx in range(len(self.transformer_layers)):
            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](x, encoder_padding_mask, attn_mask)
                if self.transformer_sampling[transformer_layer_idx] != 1:
                    sampling_factor = self.transformer_sampling[transformer_layer_idx]
                    (x, encoder_padding_mask, attn_mask) = self.slice(x, encoder_padding_mask, attn_mask, sampling_factor)
                transformer_layer_idx += 1
            else:
                x = self.transformer_layers[layer_idx](x)
        return {'encoder_out': x, 'encoder_padding_mask': encoder_padding_mask.t() if encoder_padding_mask is not None else None}

    def infer_conv_output_dim(self, in_channels, input_dim):
        if False:
            print('Hello World!')
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for (i, _) in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        x = x.transpose(1, 2)
        (mb, seq) = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def validate_transformer_config(self, transformer_config):
        if False:
            i = 10
            return i + 15
        for config in transformer_config:
            (input_dim, num_heads) = config[:2]
            if input_dim % num_heads != 0:
                msg = 'ERROR in transformer config {}: '.format(config) + 'input dimension {} '.format(input_dim) + 'not dividable by number of heads {}'.format(num_heads)
                raise ValueError(msg)

    def parse_transformer_context(self, transformer_context):
        if False:
            while True:
                i = 10
        '\n        transformer_context can be the following:\n        -   None; indicates no context is used, i.e.,\n            transformer can access full context\n        -   a tuple/list of two int; indicates left and right context,\n            any number <0 indicates infinite context\n                * e.g., (5, 6) indicates that for query at x_t, transformer can\n                access [t-5, t+6] (inclusive)\n                * e.g., (-1, 6) indicates that for query at x_t, transformer can\n                access [0, t+6] (inclusive)\n        '
        if transformer_context is None:
            return None
        if not isinstance(transformer_context, Iterable):
            raise ValueError('transformer context must be Iterable if it is not None')
        if len(transformer_context) != 2:
            raise ValueError('transformer context must have length 2')
        left_context = transformer_context[0]
        if left_context < 0:
            left_context = None
        right_context = transformer_context[1]
        if right_context < 0:
            right_context = None
        if left_context is None and right_context is None:
            return None
        return (left_context, right_context)

    def parse_transformer_sampling(self, transformer_sampling, num_layers):
        if False:
            print('Hello World!')
        '\n        parsing transformer sampling configuration\n\n        Args:\n            - transformer_sampling, accepted input:\n                * None, indicating no sampling\n                * an Iterable with int (>0) as element\n            - num_layers, expected number of transformer layers, must match with\n              the length of transformer_sampling if it is not None\n\n        Returns:\n            - A tuple with length num_layers\n        '
        if transformer_sampling is None:
            return (1,) * num_layers
        if not isinstance(transformer_sampling, Iterable):
            raise ValueError('transformer_sampling must be an iterable if it is not None')
        if len(transformer_sampling) != num_layers:
            raise ValueError('transformer_sampling {} does not match with the number of layers {}'.format(transformer_sampling, num_layers))
        for (layer, value) in enumerate(transformer_sampling):
            if not isinstance(value, int):
                raise ValueError('Invalid value in transformer_sampling: ')
            if value < 1:
                raise ValueError("{} layer's subsampling is {}.".format(layer, value) + ' This is not allowed! ')
        return transformer_sampling

    def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
        if False:
            print('Hello World!')
        '\n        embedding is a (T, B, D) tensor\n        padding_mask is a (B, T) tensor or None\n        attn_mask is a (T, T) tensor or None\n        '
        embedding = embedding[::sampling_factor, :, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::sampling_factor]
        if attn_mask is not None:
            attn_mask = attn_mask[::sampling_factor, ::sampling_factor]
        return (embedding, padding_mask, attn_mask)

    def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
        if False:
            while True:
                i = 10
        '\n        create attention mask according to sequence lengths and transformer\n        context\n\n        Args:\n            - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is\n              the length of b-th sequence\n            - subsampling_factor: int\n                * Note that the left_context and right_context is specified in\n                  the input frame-level while input to transformer may already\n                  go through subsampling (e.g., the use of striding in vggblock)\n                  we use subsampling_factor to scale the left/right context\n\n        Return:\n            - a (T, T) binary tensor or None, where T is max(input_lengths)\n                * if self.transformer_context is None, None\n                * if left_context is None,\n                    * attn_mask[t, t + right_context + 1:] = 1\n                    * others = 0\n                * if right_context is None,\n                    * attn_mask[t, 0:t - left_context] = 1\n                    * others = 0\n                * elsif\n                    * attn_mask[t, t - left_context: t + right_context + 1] = 0\n                    * others = 1\n        '
        if self.transformer_context is None:
            return None
        maxT = torch.max(input_lengths).item()
        attn_mask = torch.zeros(maxT, maxT)
        left_context = self.transformer_context[0]
        right_context = self.transformer_context[1]
        if left_context is not None:
            left_context = math.ceil(self.transformer_context[0] / subsampling_factor)
        if right_context is not None:
            right_context = math.ceil(self.transformer_context[1] / subsampling_factor)
        for t in range(maxT):
            if left_context is not None:
                st = 0
                en = max(st, t - left_context)
                attn_mask[t, st:en] = 1
            if right_context is not None:
                st = t + right_context + 1
                st = min(st, maxT - 1)
                attn_mask[t, st:] = 1
        return attn_mask.to(input_lengths.device)

    def reorder_encoder_out(self, encoder_out, new_order):
        if False:
            i = 10
            return i + 15
        encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, dictionary, embed_dim=512, transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG, conv_config=DEFAULT_DEC_CONV_CONFIG, encoder_output_dim=512):
        if False:
            print('Hello World!')
        super().__init__(dictionary)
        vocab_size = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)
        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_config)):
            (out_channels, kernel_size, layer_norm) = conv_config[i]
            if i == 0:
                conv_layer = LinearizedConv1d(embed_dim, out_channels, kernel_size, padding=kernel_size - 1)
            else:
                conv_layer = LinearizedConv1d(conv_config[i - 1][0], out_channels, kernel_size, padding=kernel_size - 1)
            self.conv_layers.append(conv_layer)
            if layer_norm:
                self.conv_layers.append(nn.LayerNorm(out_channels))
            self.conv_layers.append(nn.ReLU())
        self.layers = nn.ModuleList()
        if conv_config[-1][0] != transformer_config[0][0]:
            self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
        self.layers.append(TransformerDecoderLayer(prepare_transformer_decoder_params(*transformer_config[0])))
        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.layers.append(Linear(transformer_config[i - 1][0], transformer_config[i][0]))
            self.layers.append(TransformerDecoderLayer(prepare_transformer_decoder_params(*transformer_config[i])))
        self.fc_out = Linear(transformer_config[-1][0], vocab_size)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        if False:
            while True:
                i = 10
        "\n        Args:\n            prev_output_tokens (LongTensor): previous decoder outputs of shape\n                `(batch, tgt_len)`, for input feeding/teacher forcing\n            encoder_out (Tensor, optional): output from the encoder, used for\n                encoder-side attention\n            incremental_state (dict): dictionary used for storing state during\n                :ref:`Incremental decoding`\n        Returns:\n            tuple:\n                - the last decoder layer's output of shape `(batch, tgt_len,\n                  vocab)`\n                - the last decoder layer's attention weights of shape `(batch,\n                  tgt_len, src_len)`\n        "
        target_padding_mask = (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device) if incremental_state is None else None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self.embed_tokens(prev_output_tokens)
        x = self._transpose_if_training(x, incremental_state)
        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)
        x = self._transpose_if_inference(x, incremental_state)
        for layer in self.layers:
            if isinstance(layer, TransformerDecoderLayer):
                (x, *_) = layer(x, encoder_out['encoder_out'] if encoder_out is not None else None, encoder_out['encoder_padding_mask'].t() if encoder_out['encoder_padding_mask'] is not None else None, incremental_state, self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None, self_attn_padding_mask=target_padding_mask if incremental_state is None else None)
            else:
                x = layer(x)
        x = x.transpose(0, 1)
        x = self.fc_out(x)
        return (x, None)

    def buffered_future_mask(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def _transpose_if_training(self, x, incremental_state):
        if False:
            for i in range(10):
                print('nop')
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if False:
            print('Hello World!')
        if incremental_state:
            x = x.transpose(0, 1)
        return x

@register_model('asr_vggtransformer_encoder')
class VGGTransformerEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        if False:
            i = 10
            return i + 15
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        if False:
            while True:
                i = 10
        'Add model-specific arguments to the parser.'
        parser.add_argument('--input-feat-per-channel', type=int, metavar='N', help='encoder input dimension per input channel')
        parser.add_argument('--vggblock-enc-config', type=str, metavar='EXPR', help='\n    an array of tuples each containing the configuration of one vggblock\n    [(out_channels, conv_kernel_size, pooling_kernel_size,num_conv_layers), ...]\n    ')
        parser.add_argument('--transformer-enc-config', type=str, metavar='EXPR', help='\n    a tuple containing the configuration of the Transformer layers\n    configurations:\n    [(input_dim,\n      num_heads,\n      ffn_dim,\n      normalize_before,\n      dropout,\n      attention_dropout,\n      relu_dropout), ]')
        parser.add_argument('--enc-output-dim', type=int, metavar='N', help='encoder output dimension, projecting the LSTM output')
        parser.add_argument('--in-channels', type=int, metavar='N', help='number of encoder input channels')
        parser.add_argument('--transformer-context', type=str, metavar='EXPR', help='\n    either None or a tuple of two ints, indicating left/right context a\n    transformer can have access to')
        parser.add_argument('--transformer-sampling', type=str, metavar='EXPR', help='\n    either None or a tuple of ints, indicating sampling factor in each layer')

    @classmethod
    def build_model(cls, args, task):
        if False:
            while True:
                i = 10
        'Build a new model instance.'
        base_architecture_enconly(args)
        encoder = VGGTransformerEncoderOnly(vocab_size=len(task.target_dictionary), input_feat_per_channel=args.input_feat_per_channel, vggblock_config=eval(args.vggblock_enc_config), transformer_config=eval(args.transformer_enc_config), encoder_output_dim=args.enc_output_dim, in_channels=args.in_channels, transformer_context=eval(args.transformer_context), transformer_sampling=eval(args.transformer_sampling))
        return cls(encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            i = 10
            return i + 15
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs = lprobs.transpose(0, 1).contiguous()
        lprobs.batch_first = True
        return lprobs

class VGGTransformerEncoderOnly(VGGTransformerEncoder):

    def __init__(self, vocab_size, input_feat_per_channel, vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG, transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG, encoder_output_dim=512, in_channels=1, transformer_context=None, transformer_sampling=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(input_feat_per_channel=input_feat_per_channel, vggblock_config=vggblock_config, transformer_config=transformer_config, encoder_output_dim=encoder_output_dim, in_channels=in_channels, transformer_context=transformer_context, transformer_sampling=transformer_sampling)
        self.fc_out = Linear(self.encoder_output_dim, vocab_size)

    def forward(self, src_tokens, src_lengths, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        src_tokens: padded tensor (B, T, C * feat)\n        src_lengths: tensor of original lengths of input utterances (B,)\n        '
        enc_out = super().forward(src_tokens, src_lengths)
        x = self.fc_out(enc_out['encoder_out'])
        return {'encoder_out': x, 'encoder_padding_mask': enc_out['encoder_padding_mask']}

    def max_positions(self):
        if False:
            print('Hello World!')
        'Maximum input length supported by the encoder.'
        return (1000000.0, 1000000.0)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    if False:
        i = 10
        return i + 15
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    return m

def Linear(in_features, out_features, bias=True, dropout=0):
    if False:
        for i in range(10):
            print('nop')
    'Linear layer (input: N x T x C)'
    m = nn.Linear(in_features, out_features, bias=bias)
    return m

def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    if False:
        print('Hello World!')
    'Weight-normalized Conv1d layer optimized for decoding'
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(4 * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

def LayerNorm(embedding_dim):
    if False:
        return 10
    m = nn.LayerNorm(embedding_dim)
    return m

def base_architecture(args):
    if False:
        print('Hello World!')
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 40)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', DEFAULT_ENC_VGGBLOCK_CONFIG)
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', DEFAULT_ENC_TRANSFORMER_CONFIG)
    args.enc_output_dim = getattr(args, 'enc_output_dim', 512)
    args.in_channels = getattr(args, 'in_channels', 1)
    args.tgt_embed_dim = getattr(args, 'tgt_embed_dim', 128)
    args.transformer_dec_config = getattr(args, 'transformer_dec_config', DEFAULT_ENC_TRANSFORMER_CONFIG)
    args.conv_dec_config = getattr(args, 'conv_dec_config', DEFAULT_DEC_CONV_CONFIG)
    args.transformer_context = getattr(args, 'transformer_context', 'None')

@register_model_architecture('asr_vggtransformer', 'vggtransformer_1')
def vggtransformer_1(args):
    if False:
        print('Hello World!')
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 80)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', '[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]')
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', '((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 14')
    args.enc_output_dim = getattr(args, 'enc_output_dim', 1024)
    args.tgt_embed_dim = getattr(args, 'tgt_embed_dim', 128)
    args.conv_dec_config = getattr(args, 'conv_dec_config', '((256, 3, True),) * 4')
    args.transformer_dec_config = getattr(args, 'transformer_dec_config', '((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 4')

@register_model_architecture('asr_vggtransformer', 'vggtransformer_2')
def vggtransformer_2(args):
    if False:
        i = 10
        return i + 15
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 80)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', '[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]')
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', '((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16')
    args.enc_output_dim = getattr(args, 'enc_output_dim', 1024)
    args.tgt_embed_dim = getattr(args, 'tgt_embed_dim', 512)
    args.conv_dec_config = getattr(args, 'conv_dec_config', '((256, 3, True),) * 4')
    args.transformer_dec_config = getattr(args, 'transformer_dec_config', '((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 6')

@register_model_architecture('asr_vggtransformer', 'vggtransformer_base')
def vggtransformer_base(args):
    if False:
        return 10
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 80)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', '[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]')
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', '((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 12')
    args.enc_output_dim = getattr(args, 'enc_output_dim', 512)
    args.tgt_embed_dim = getattr(args, 'tgt_embed_dim', 512)
    args.conv_dec_config = getattr(args, 'conv_dec_config', '((256, 3, True),) * 4')
    args.transformer_dec_config = getattr(args, 'transformer_dec_config', '((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6')

def base_architecture_enconly(args):
    if False:
        return 10
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 40)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', '[(32, 3, 2, 2, True)] * 2')
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', '((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2')
    args.enc_output_dim = getattr(args, 'enc_output_dim', 512)
    args.in_channels = getattr(args, 'in_channels', 1)
    args.transformer_context = getattr(args, 'transformer_context', 'None')
    args.transformer_sampling = getattr(args, 'transformer_sampling', 'None')

@register_model_architecture('asr_vggtransformer_encoder', 'vggtransformer_enc_1')
def vggtransformer_enc_1(args):
    if False:
        return 10
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 80)
    args.vggblock_enc_config = getattr(args, 'vggblock_enc_config', '[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]')
    args.transformer_enc_config = getattr(args, 'transformer_enc_config', '((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16')
    args.enc_output_dim = getattr(args, 'enc_output_dim', 1024)