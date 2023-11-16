import re
from dataclasses import dataclass, field, fields
from typing import List, Optional
from omegaconf import II
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.utils import safe_getattr, safe_hasattr
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(100000000.0)
_NAME_PARSER = '(decoder|encoder|quant_noise)_(.*)'

@dataclass
class EncDecBaseConfig(FairseqDataclass):
    embed_path: Optional[str] = field(default=None, metadata={'help': 'path to pre-trained embedding'})
    embed_dim: Optional[int] = field(default=512, metadata={'help': 'embedding dimension'})
    ffn_embed_dim: int = field(default=2048, metadata={'help': 'embedding dimension for FFN'})
    layers: int = field(default=6, metadata={'help': 'number of layers'})
    attention_heads: int = field(default=8, metadata={'help': 'number of attention heads'})
    normalize_before: bool = field(default=False, metadata={'help': 'apply layernorm before each block'})
    learned_pos: bool = field(default=False, metadata={'help': 'use learned positional embeddings'})
    layerdrop: float = field(default=0, metadata={'help': 'LayerDrop probability'})
    layers_to_keep: Optional[List[int]] = field(default=None, metadata={'help': 'which layers to *keep* when pruning'})
    xformers_att_config: Optional[str] = field(default=None, metadata={'help': 'config for xFormers attention, defined in xformers.components.attention.AttentionConfig'})

@dataclass
class DecoderConfig(EncDecBaseConfig):
    input_dim: int = II('model.decoder.embed_dim')
    output_dim: int = field(default=II('model.decoder.embed_dim'), metadata={'help': 'decoder output dimension (extra linear layer if different from decoder embed dim)'})

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.input_dim == II('model.decoder.embed_dim'):
            self.input_dim = self.embed_dim
        if self.output_dim == II('model.decoder.embed_dim'):
            self.output_dim = self.embed_dim

@dataclass
class QuantNoiseConfig(FairseqDataclass):
    pq: float = field(default=0.0, metadata={'help': 'iterative PQ quantization noise at training time'})
    pq_block_size: int = field(default=8, metadata={'help': 'block size of quantization noise at training time'})
    scalar: float = field(default=0.0, metadata={'help': 'scalar quantization noise and scalar quantization at training time'})

@dataclass
class TransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='relu', metadata={'help': 'activation function to use'})
    dropout: float = field(default=0.1, metadata={'help': 'dropout probability'})
    attention_dropout: float = field(default=0.0, metadata={'help': 'dropout probability for attention weights'})
    activation_dropout: float = field(default=0.0, metadata={'help': 'dropout probability after activation in FFN.', 'alias': '--relu-dropout'})
    adaptive_input: bool = False
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    max_source_positions: int = field(default=DEFAULT_MAX_SOURCE_POSITIONS, metadata={'help': 'Maximum input length supported by the encoder'})
    decoder: DecoderConfig = DecoderConfig()
    max_target_positions: int = field(default=DEFAULT_MAX_TARGET_POSITIONS, metadata={'help': 'Maximum output length supported by the decoder'})
    share_decoder_input_output_embed: bool = field(default=False, metadata={'help': 'share decoder input and output embeddings'})
    share_all_embeddings: bool = field(default=False, metadata={'help': 'share encoder, decoder and output embeddings (requires shared dictionary and embed dim)'})
    merge_src_tgt_embed: bool = field(default=False, metadata={'help': 'if true then the source and target embedding table is merged into one table. This is going to make the model smaller but it might hurt performance.'})
    no_token_positional_embeddings: bool = field(default=False, metadata={'help': 'if True, disables positional embeddings (outside self attention)'})
    adaptive_softmax_cutoff: Optional[List[int]] = field(default=None, metadata={'help': 'list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion'})
    adaptive_softmax_dropout: float = field(default=0.0, metadata={'help': 'sets adaptive softmax dropout for the tail projections'})
    adaptive_softmax_factor: float = field(default=4, metadata={'help': 'adaptive input factor'})
    layernorm_embedding: bool = field(default=False, metadata={'help': 'add layernorm to embedding'})
    tie_adaptive_weights: bool = field(default=False, metadata={'help': 'if set, ties the weights of adaptive softmax and adaptive input'})
    tie_adaptive_proj: bool = field(default=False, metadata={'help': 'if set, ties the projection weights of adaptive softmax and adaptive input'})
    no_scale_embedding: bool = field(default=False, metadata={'help': 'if True, dont scale embeddings'})
    checkpoint_activations: bool = field(default=False, metadata={'help': 'checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute'})
    offload_activations: bool = field(default=False, metadata={'help': 'checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.'})
    no_cross_attention: bool = field(default=False, metadata={'help': 'do not perform cross-attention'})
    cross_self_attention: bool = field(default=False, metadata={'help': 'perform cross+self-attention'})
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(default=DEFAULT_MIN_PARAMS_TO_WRAP, metadata={'help': 'minimum number of params for a layer to be wrapped with FSDP() when training with --ddp-backend=fully_sharded. Smaller values will improve memory efficiency, but may make torch.distributed communication less efficient due to smaller input sizes. This option is set to 0 (i.e., always wrap) when --checkpoint-activations or --offload-activations are passed.'})
    char_inputs: bool = field(default=False, metadata={'help': 'if set, model takes character ids as input'})
    relu_dropout: float = 0.0
    base_layers: Optional[int] = field(default=0, metadata={'help': 'number of BASE layers in total'})
    base_sublayers: Optional[int] = field(default=1, metadata={'help': 'number of sublayers in each BASE layer'})
    base_shuffle: Optional[int] = field(default=1, metadata={'help': 'shuffle tokens between workers before computing assignment'})
    export: bool = field(default=False, metadata={'help': 'make the layernorm exportable with torchscript.'})
    no_decoder_final_norm: bool = field(default=False, metadata={'help': "don't add an extra layernorm after the last decoder block"})

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            return safe_getattr(sub, match[2])
        raise AttributeError(f'invalid argument {name}.')

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        match = re.match(_NAME_PARSER, name)
        if match:
            sub = safe_getattr(self, match[1])
            setattr(sub, match[2], value)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _copy_keys(args, cls, prefix, seen):
        if False:
            while True:
                i = 10
        '\n        copy the prefixed keys (decoder_embed_dim) to the DC fields: decoder.embed_dim\n        '
        cfg = cls()
        for fld in fields(cls):
            args_key = f'{prefix}_{fld.name}'
            if safe_hasattr(args, args_key):
                seen.add(args_key)
                setattr(cfg, fld.name, safe_getattr(args, args_key))
            if safe_hasattr(args, fld.name):
                seen.add(fld.name)
                setattr(cfg, fld.name, safe_getattr(args, fld.name))
        return cfg

    @classmethod
    def from_namespace(cls, args):
        if False:
            return 10
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            for fld in fields(cls):
                if fld.name == 'decoder':
                    if safe_hasattr(args, 'decoder'):
                        seen.add('decoder')
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(args, DecoderConfig, 'decoder', seen)
                elif fld.name == 'encoder':
                    if safe_hasattr(args, 'encoder'):
                        seen.add('encoder')
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(args, EncDecBaseConfig, 'encoder', seen)
                elif fld.name == 'quant_noise':
                    if safe_hasattr(args, 'quant_noise'):
                        seen.add('quant_noise')
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(args, QuantNoiseConfig, 'quant_noise', seen)
                elif safe_hasattr(args, fld.name):
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            args_dict = args._asdict() if safe_hasattr(args, '_asdict') else vars(args) if safe_hasattr(args, '__dict__') else {}
            for (key, value) in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args