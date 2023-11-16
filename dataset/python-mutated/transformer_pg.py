import logging
from typing import Any, Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, TransformerDecoder, TransformerEncoder, TransformerModel, base_architecture
from torch import Tensor
logger = logging.getLogger(__name__)

@register_model('transformer_pointer_generator')
class TransformerPointerGeneratorModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_, augmented with a pointer-generator
    network from `"Get To The Point: Summarization with Pointer-Generator
    Networks" (See et al, 2017) <https://arxiv.org/abs/1704.04368>`_.

    Args:
        encoder (TransformerPointerGeneratorEncoder): the encoder
        decoder (TransformerPointerGeneratorDecoder): the decoder

    The Transformer pointer-generator model provides the following named
    architectures and command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_pointer_generator_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        'Add model-specific arguments to the parser.'
        TransformerModel.add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='N', help='number of attention heads to be used for pointing')
        parser.add_argument('--alignment-layer', type=int, metavar='I', help='layer number to be used for pointing (0 corresponding to the bottommost layer)')
        parser.add_argument('--source-position-markers', type=int, metavar='N', help='dictionary includes N additional items that represent an OOV token at a particular input position')
        parser.add_argument('--force-generation', type=float, metavar='P', default=None, help='set the vocabulary distribution weight to P, instead of predicting it from the input (1.0 corresponding to generation, 0.0 to pointing)')

    @classmethod
    def build_model(cls, args, task):
        if False:
            return 10
        'Build a new model instance.'
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(','))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(','))
        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        if getattr(args, 'source_position_markers', None) is None:
            args.source_position_markers = args.max_source_positions
        (src_dict, tgt_dict) = (task.source_dictionary, task.target_dictionary)
        if src_dict != tgt_dict:
            raise ValueError('Pointer-generator requires a joined dictionary')

        def build_embedding(dictionary, embed_dim, path=None):
            if False:
                for i in range(10):
                    print('nop')
            num_embeddings = len(dictionary) - args.source_position_markers
            padding_idx = dictionary.pad()
            unk_idx = dictionary.unk()
            logger.info('dictionary indices from {0} to {1} will be mapped to {2}'.format(num_embeddings, len(dictionary) - 1, unk_idx))
            emb = Embedding(num_embeddings, embed_dim, padding_idx, unk_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError('--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and args.decoder_embed_path != args.encoder_embed_path:
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if False:
            i = 10
            return i + 15
        return TransformerPointerGeneratorEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if False:
            print('Hello World!')
        return TransformerPointerGeneratorDecoder(args, tgt_dict, embed_tokens)

class TransformerPointerGeneratorEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`. The pointer-generator variant adds
    the source tokens to the encoder output as these are otherwise not passed
    to the decoder.
    """

    def forward(self, src_tokens, src_lengths: Optional[Tensor]=None, return_all_hiddens: bool=False, token_embeddings: Optional[Tensor]=None):
        if False:
            return 10
        "\n        Runs the `forward()` method of the parent Transformer class. Then adds\n        the source tokens into the encoder output tuple.\n\n        While it might be more elegant that the model would pass the source\n        tokens to the `forward()` method of the decoder too, this would require\n        changes to `SequenceGenerator`.\n\n        Args:\n            src_tokens (torch.LongTensor): tokens in the source language of\n                shape `(batch, src_len)`\n            src_lengths (torch.LongTensor): lengths of each source sentence of\n                shape `(batch)`\n            return_all_hiddens (bool, optional): also return all of the\n                intermediate hidden states (default: False).\n            token_embeddings (torch.Tensor, optional): precomputed embeddings\n                default `None` will recompute embeddings\n\n        Returns:\n            namedtuple:\n                - **encoder_out** (Tensor): the last encoder layer's output of\n                  shape `(src_len, batch, embed_dim)`\n                - **encoder_padding_mask** (ByteTensor): the positions of\n                  padding elements of shape `(batch, src_len)`\n                - **encoder_embedding** (Tensor): the (scaled) embedding lookup\n                  of shape `(batch, src_len, embed_dim)`\n                - **encoder_states** (List[Tensor]): all intermediate\n                  hidden states of shape `(src_len, batch, embed_dim)`.\n                  Only populated if *return_all_hiddens* is True.\n                - **src_tokens** (Tensor): input token ids of shape\n                  `(batch, src_len)`\n        "
        encoder_out = self.forward_scriptable(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        return {'encoder_out': encoder_out['encoder_out'], 'encoder_padding_mask': encoder_out['encoder_padding_mask'], 'encoder_embedding': encoder_out['encoder_embedding'], 'encoder_states': encoder_out['encoder_states'], 'src_tokens': [src_tokens], 'src_lengths': []}

class TransformerPointerGeneratorDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`. The pointer-generator variant mixes
    the output probabilities with an attention distribution in the output layer.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        if False:
            print('Hello World!')
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        input_embed_dim = embed_tokens.embedding_dim
        p_gen_input_size = input_embed_dim + self.output_embed_dim
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        nn.init.zeros_(self.project_p_gens.bias)
        self.num_types = len(dictionary)
        self.num_oov_types = args.source_position_markers
        self.num_embeddings = self.num_types - self.num_oov_types
        self.force_p_gen = args.force_generation

    def forward(self, prev_output_tokens, encoder_out: Optional[Dict[str, List[Tensor]]]=None, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None, features_only: bool=False, alignment_layer: Optional[int]=0, alignment_heads: Optional[int]=1, src_lengths: Optional[Any]=None, return_all_hiddens: bool=False):
        if False:
            while True:
                i = 10
        "\n        Args:\n            prev_output_tokens (LongTensor): previous decoder outputs of shape\n                `(batch, tgt_len)`, for teacher forcing\n            encoder_out (optional): output from the encoder, used for\n                encoder-side attention\n            incremental_state (dict, optional): dictionary used for storing\n                state during :ref:`Incremental decoding`\n            features_only (bool, optional): only return features without\n                applying output layer (default: False)\n            alignment_layer (int, optional): 0-based index of the layer to be\n                used for pointing (default: 0)\n            alignment_heads (int, optional): number of attention heads to be\n                used for pointing (default: 1)\n\n        Returns:\n            tuple:\n                - the decoder's output of shape `(batch, tgt_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        (x, extra) = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, alignment_layer=self.alignment_layer, alignment_heads=self.alignment_heads)
        if not features_only:
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, x), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens.float())
            attn: Optional[Tensor] = extra['attn'][0]
            assert encoder_out is not None
            assert attn is not None
            x = self.output_layer(x, attn, encoder_out['src_tokens'][0], p_gens)
        return (x, extra)

    def output_layer(self, features: Tensor, attn: Tensor, src_tokens: Tensor, p_gens: Tensor) -> Tensor:
        if False:
            return 10
        '\n        Project features to the vocabulary size and mix with the attention\n        distributions.\n        '
        if self.force_p_gen is not None:
            p_gens = self.force_p_gen
        if self.adaptive_softmax is None:
            logits = self.output_projection(features)
        else:
            logits = features
        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert logits.shape[2] == self.num_embeddings
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]
        gen_dists = self.get_normalized_probs_scriptable((logits, None), log_probs=False, sample=None)
        gen_dists = torch.mul(gen_dists, p_gens)
        padding_size = (batch_size, output_length, self.num_oov_types)
        padding = gen_dists.new_zeros(padding_size)
        gen_dists = torch.cat((gen_dists, padding), 2)
        assert gen_dists.shape[2] == self.num_types
        attn = torch.mul(attn.float(), 1 - p_gens)
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        attn_dists = attn.new_zeros(attn_dists_size)
        attn_dists.scatter_add_(2, index, attn.float())
        return gen_dists + attn_dists

    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get normalized probabilities (or log probs) from a net's output.\n        Pointer-generator network output is already normalized.\n        "
        probs = net_output[0]
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs

class Embedding(nn.Embedding):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings. This subclass differs from the standard PyTorch Embedding class by
    allowing additional vocabulary entries that will be mapped to the unknown token
    embedding.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int): Pads the output with the embedding vector at :attr:`padding_idx`
                           (initialized to zeros) whenever it encounters the index.
        unk_idx (int): Maps all token indices that are greater than or equal to
                       num_embeddings to this index.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\\text{embedding\\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    """
    __constants__ = ['unk_idx']

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int], unk_idx: int, max_norm: Optional[float]=float('inf')):
        if False:
            print('Hello World!')
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)
        self.unk_idx = unk_idx
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** (-0.5))
        nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        input = torch.where(input >= self.num_embeddings, torch.ones_like(input) * self.unk_idx, input)
        return nn.functional.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator')
def transformer_pointer_generator(args):
    if False:
        for i in range(10):
            print('nop')
    args.alignment_heads = getattr(args, 'alignment_heads', 1)
    args.alignment_layer = getattr(args, 'alignment_layer', -1)
    base_architecture(args)
    if args.alignment_layer < 0:
        args.alignment_layer = args.decoder_layers + args.alignment_layer

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_iwslt_de_en')
def transformer_pointer_generator_iwslt_de_en(args):
    if False:
        for i in range(10):
            print('nop')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    transformer_pointer_generator(args)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_wmt_en_de')
def transformer_pointer_generator_wmt_en_de(args):
    if False:
        for i in range(10):
            print('nop')
    transformer_pointer_generator(args)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_vaswani_wmt_en_de_big')
def transformer_pointer_generator_vaswani_wmt_en_de_big(args):
    if False:
        return 10
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    transformer_pointer_generator(args)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_vaswani_wmt_en_fr_big')
def transformer_pointer_generator_vaswani_wmt_en_fr_big(args):
    if False:
        while True:
            i = 10
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_pointer_generator_vaswani_wmt_en_de_big(args)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_wmt_en_de_big')
def transformer_pointer_generator_wmt_en_de_big(args):
    if False:
        i = 10
        return i + 15
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_pointer_generator_vaswani_wmt_en_de_big(args)

@register_model_architecture('transformer_pointer_generator', 'transformer_pointer_generator_wmt_en_de_big_t2t')
def transformer_pointer_generator_wmt_en_de_big_t2t(args):
    if False:
        while True:
            i = 10
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_pointer_generator_vaswani_wmt_en_de_big(args)