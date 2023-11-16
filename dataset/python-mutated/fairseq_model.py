"""
Base classes for various fairseq models.
"""
import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, gen_parser_from_dataclass
from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig
from torch import Tensor
logger = logging.getLogger(__name__)

def check_type(module, expected_type):
    if False:
        i = 10
        return i + 15
    if hasattr(module, 'unwrapped_module'):
        assert isinstance(module.unwrapped_module, expected_type), f'{type(module.unwrapped_module)} != {expected_type}'
    else:
        assert isinstance(module, expected_type), f'{type(module)} != {expected_type}'

class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._is_generation_fast = False

    @classmethod
    def add_args(cls, parser):
        if False:
            i = 10
            return i + 15
        'Add model-specific arguments to the parser.'
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_model(cls, args, task):
        if False:
            while True:
                i = 10
        'Build a new model instance.'
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        if False:
            for i in range(10):
                print('nop')
        "Get targets from either the sample or the net's output."
        return sample['target']

    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            while True:
                i = 10
        "Get normalized probabilities (or log probs) from a net's output."
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]], log_probs: bool, sample: Optional[Dict[str, Tensor]]=None):
        if False:
            return 10
        'Scriptable helper function for get_normalized_probs in ~BaseFairseqModel'
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Similar to *forward* but only return features.'
        return self(*args, **kwargs)

    def max_positions(self):
        if False:
            while True:
                i = 10
        'Maximum length supported by the model.'
        return None

    def load_state_dict(self, state_dict, strict=True, model_cfg: Optional[DictConfig]=None, args: Optional[Namespace]=None):
        if False:
            return 10
        'Copies parameters and buffers from *state_dict* into this module and\n        its descendants.\n\n        Overrides the method in :class:`nn.Module`. Compared with that method\n        this additionally "upgrades" *state_dicts* from old checkpoints.\n        '
        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model
        self.upgrade_state_dict(state_dict)
        from fairseq.checkpoint_utils import prune_state_dict
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        if False:
            i = 10
            return i + 15
        'Upgrade old state dicts to work with newer code.'
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            print('Hello World!')
        'Upgrade old state dicts to work with newer code.\n\n        Args:\n            state_dict (dict): state dictionary to upgrade, in place\n            name (str): the state dict key corresponding to the current module\n        '
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if False:
                i = 10
                return i + 15
            if len(prefix) > 0:
                prefix += '.'
            for (n, c) in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)
        do_upgrade(self, name)

    def set_num_updates(self, num_updates):
        if False:
            while True:
                i = 10
        'State from trainer to pass along to model at every update.'
        for m in self.modules():
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)

    def set_epoch(self, epoch):
        if False:
            while True:
                i = 10
        for m in self.modules():
            if hasattr(m, 'set_epoch') and m != self:
                m.set_epoch(epoch)

    def prepare_for_inference_(self, cfg: DictConfig):
        if False:
            return 10
        'Prepare model for inference.'
        kwargs = {}
        kwargs['beamable_mm_beam_size'] = None if getattr(cfg.generation, 'no_beamable_mm', False) else getattr(cfg.generation, 'beam', 5)
        kwargs['need_attn'] = getattr(cfg.generation, 'print_alignment', False)
        if getattr(cfg.generation, 'retain_dropout', False):
            kwargs['retain_dropout'] = cfg.generation.retain_dropout
            kwargs['retain_dropout_modules'] = cfg.generation.retain_dropout_modules
        self.make_generation_fast_(**kwargs)

    def make_generation_fast_(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Legacy entry point to optimize model for faster generation.\n        Prefer prepare_for_inference_.\n        '
        if self._is_generation_fast:
            return
        self._is_generation_fast = True

        def apply_remove_weight_norm(module):
            if False:
                i = 10
                return i + 15
            try:
                nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):
                return
        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module, prefix):
            if False:
                print('Hello World!')
            if len(prefix) > 0:
                prefix += '.'
            base_func = BaseFairseqModel.make_generation_fast_
            for (n, m) in module.named_modules():
                if m != self and hasattr(m, 'make_generation_fast_') and (m.make_generation_fast_.__func__ is not base_func):
                    name = prefix + n
                    m.make_generation_fast_(name=name, **kwargs)
        apply_make_generation_fast_(self, '')

        def train(mode=True):
            if False:
                print('Hello World!')
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Make model exportable via ONNX trace.'
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if False:
                for i in range(10):
                    print('nop')
            if module != self and hasattr(module, 'prepare_for_onnx_export_') and (module not in seen):
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)
        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs):
        if False:
            print('Hello World!')
        "\n        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model\n        file. Downloads and caches the pre-trained model file if needed.\n\n        The base implementation returns a\n        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to\n        generate translations or sample from language models. The underlying\n        :class:`~fairseq.models.FairseqModel` can be accessed via the\n        *generator.models* attribute.\n\n        Other models may override this to implement custom hub interfaces.\n\n        Args:\n            model_name_or_path (str): either the name of a pre-trained model to\n                load or a path/URL to a pre-trained model state dict\n            checkpoint_file (str, optional): colon-separated list of checkpoint\n                files in the model archive to ensemble (default: 'model.pt')\n            data_name_or_path (str, optional): point args.data to the archive\n                at the given path/URL. Can start with '.' or './' to reuse the\n                model archive path.\n        "
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(model_name_or_path, checkpoint_file, data_name_or_path, archive_map=cls.hub_models(), **kwargs)
        logger.info(x['args'])
        return hub_utils.GeneratorHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def hub_models(cls):
        if False:
            print('Hello World!')
        return {}

class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        check_type(self.encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if False:
            print('Hello World!')
        "\n        Run the forward pass for an encoder-decoder model.\n\n        First feed a batch of source tokens through the encoder. Then, feed the\n        encoder output and previous decoder outputs (i.e., teacher forcing) to\n        the decoder to produce the next outputs::\n\n            encoder_out = self.encoder(src_tokens, src_lengths)\n            return self.decoder(prev_output_tokens, encoder_out)\n\n        Args:\n            src_tokens (LongTensor): tokens in the source language of shape\n                `(batch, src_len)`\n            src_lengths (LongTensor): source sentence lengths of shape `(batch)`\n            prev_output_tokens (LongTensor): previous decoder outputs of shape\n                `(batch, tgt_len)`, for teacher forcing\n\n        Returns:\n            tuple:\n                - the decoder's output of shape `(batch, tgt_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        if False:
            while True:
                i = 10
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Similar to *forward* but only return features.\n\n        Returns:\n            tuple:\n                - the decoder's features of shape `(batch, tgt_len, embed_dim)`\n                - a dictionary with any model-specific outputs\n        "
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Project features to the default output size (typically vocabulary size).'
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        if False:
            for i in range(10):
                print('nop')
        'Maximum length supported by the model.'
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        if False:
            return 10
        'Maximum length supported by the decoder.'
        return self.decoder.max_positions()

class FairseqModel(FairseqEncoderDecoderModel):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        utils.deprecation_warning('FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead', stacklevel=4)

class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""

    def __init__(self, encoders, decoders):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            check_type(encoders[key], FairseqEncoder)
            check_type(decoders[key], FairseqDecoder)
        self.models = nn.ModuleDict({key: FairseqEncoderDecoderModel(encoders[key], decoders[key]) for key in self.keys})

    @staticmethod
    def build_shared_embeddings(dicts: Dict[str, Dictionary], langs: List[str], embed_dim: int, build_embedding: callable, pretrained_embed_path: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n        Helper function to build shared embeddings for a set of languages after\n        checking that all dicts corresponding to those languages are equivalent.\n\n        Args:\n            dicts: Dict of lang_id to its corresponding Dictionary\n            langs: languages that we want to share embeddings for\n            embed_dim: embedding dimension\n            build_embedding: callable function to actually build the embedding\n            pretrained_embed_path: Optional path to load pretrained embeddings\n        '
        shared_dict = dicts[langs[0]]
        if any((dicts[lang] != shared_dict for lang in langs)):
            raise ValueError('--share-*-embeddings requires a joined dictionary: --share-encoder-embeddings requires a joined source dictionary, --share-decoder-embeddings requires a joined target dictionary, and --share-all-embeddings requires a joint source + target dictionary.')
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    def max_positions(self):
        if False:
            for i in range(10):
                print('nop')
        'Maximum length supported by the model.'
        return {key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions()) for key in self.keys}

    def max_decoder_positions(self):
        if False:
            print('Hello World!')
        'Maximum length supported by the decoder.'
        return min((model.decoder.max_positions() for model in self.models.values()))

    @property
    def encoder(self):
        if False:
            return 10
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.models[self.keys[0]].decoder

    def forward_decoder(self, prev_output_tokens, **kwargs):
        if False:
            while True:
                i = 10
        return self.decoder(prev_output_tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args: Optional[Namespace]=None):
        if False:
            return 10
        'Copies parameters and buffers from *state_dict* into this module and\n        its descendants.\n\n        Overrides the method in :class:`nn.Module`. Compared with that method\n        this additionally "upgrades" *state_dicts* from old checkpoints.\n        '
        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model
        self.upgrade_state_dict(state_dict)
        from fairseq.checkpoint_utils import prune_state_dict
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        if False:
            return 10
        super().__init__()
        self.decoder = decoder
        check_type(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Run the forward pass for a decoder-only model.\n\n        Feeds a batch of tokens through the decoder to predict the next tokens.\n\n        Args:\n            src_tokens (LongTensor): tokens on which to condition the decoder,\n                of shape `(batch, tgt_len)`\n            src_lengths (LongTensor): source sentence lengths of shape `(batch)`\n\n        Returns:\n            tuple:\n                - the decoder's output of shape `(batch, seq_len, vocab)`\n                - a dictionary with any model-specific outputs\n        "
        return self.decoder(src_tokens, **kwargs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        if False:
            print('Hello World!')
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, **kwargs):
        if False:
            return 10
        "\n        Similar to *forward* but only return features.\n\n        Returns:\n            tuple:\n                - the decoder's features of shape `(batch, seq_len, embed_dim)`\n                - a dictionary with any model-specific outputs\n        "
        return self.decoder.extract_features(src_tokens, **kwargs)

    def output_layer(self, features, **kwargs):
        if False:
            return 10
        'Project features to the default output size (typically vocabulary size).'
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        if False:
            while True:
                i = 10
        'Maximum length supported by the model.'
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        if False:
            i = 10
            return i + 15
        'Maximum length supported by the decoder.'
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        if False:
            return 10
        return {'future'}

class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        if False:
            print('Hello World!')
        super().__init__()
        self.encoder = encoder
        check_type(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Run the forward pass for a encoder-only model.\n\n        Feeds a batch of tokens through the encoder to generate features.\n\n        Args:\n            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`\n            src_lengths (LongTensor): source sentence lengths of shape `(batch)`\n\n        Returns:\n            the encoder's output, typically of shape `(batch, src_len, features)`\n        "
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            for i in range(10):
                print('nop')
        "Get normalized probabilities (or log probs) from a net's output."
        encoder_out = net_output['encoder_out']
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        if False:
            return 10
        'Maximum length supported by the model.'
        return self.encoder.max_positions()