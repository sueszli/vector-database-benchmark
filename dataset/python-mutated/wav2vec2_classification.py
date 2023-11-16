import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES, Wav2Vec2Config
from fairseq.models.wav2vec.wav2vec2_asr import Embedding, Linear, Wav2VecEncoder, Wav2Vec2AsrConfig
from fairseq.tasks import FairseqTask
logging.basicConfig(level=logging.DEBUG)

@dataclass
class Wav2Vec2ClassificationConfig(Wav2Vec2AsrConfig):
    latent_embed_dim: Optional[int] = field(default=None, metadata={'help': 'latent dim (encoder w2v -> latent -> class'})
    pooling: str = field(default='first_token', metadata={'help': 'pooling layer choices'})
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='gelu', metadata={'help': 'activation function to use'})

@register_model('wav2vec_classification', dataclass=Wav2Vec2ClassificationConfig)
class Wav2VecClassification(BaseFairseqModel):

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, w2v_encoder: BaseFairseqModel, pooling_layer):
        if False:
            print('Hello World!')
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.pooling_layer = pooling_layer

    def upgrade_state_dict_named(self, state_dict, name):
        if False:
            return 10
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2ClassificationConfig, task: FairseqTask):
        if False:
            return 10
        'Build a new model instance.'
        w2v_encoder = Wav2VecEncoder(cfg, None)
        pooling_layer = get_pooling_layer(cfg, w2v_encoder.w2v_model.encoder.layers[-1].embedding_dim, len(task.target_dictionary), len(w2v_encoder.w2v_model.encoder.layers))
        return cls(cfg, w2v_encoder, pooling_layer)

    def get_normalized_probs(self, net_output, log_probs):
        if False:
            print('Hello World!')
        "Get normalized probabilities (or log probs) from a net's output."
        logits = net_output
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        if False:
            print('Hello World!')
        return net_output

    def forward(self, **kwargs):
        if False:
            return 10
        encoder_out_dict = self.w2v_encoder(**kwargs)
        w2v_encoder_out = encoder_out_dict['encoder_out']
        w2v_encoder_padding_mask = encoder_out_dict['padding_mask']
        return self.pooling_layer(last_layer_feats=w2v_encoder_out, padding_mask=w2v_encoder_padding_mask)

def get_pooling_layer(cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int, encoder_layers: int):
    if False:
        for i in range(10):
            print('nop')
    assert cfg.pooling == 'mean'
    if cfg.pooling == 'first_token':
        return FirstToken(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == 'mean':
        return MeanPoolingFast(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == 'mean_amsoftmax':
        return MeanPoolingFastAMSoftmax(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == 'max':
        return MaxPoolingFast(cfg, encoder_embed_dim, num_targets)
    elif cfg.pooling == 'elmo':
        return LayerWeightedMeanPooling(cfg, encoder_embed_dim, num_targets, encoder_layers)
    else:
        raise NotImplementedError(f'{cfg.pooling} has not been implemented yet.')

class Pooling(nn.Module):

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.projection = Linear(encoder_embed_dim, num_targets)

    def forward(self, last_layer_feats, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError()

class FirstToken(Pooling):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

    def forward(self, last_layer_feats, **kwargs):
        if False:
            while True:
                i = 10
        return self.projection(last_layer_feats[:, 0])

def fn_mean(x, mask):
    if False:
        for i in range(10):
            print('nop')
    '\n    Args:\n        x: TxBxD\n        mask: BxT\n    Return:\n        y: BxD\n    '
    if mask is not None:
        mask = mask.t()[:, :, None]
        return (x * mask).sum(0) / mask.sum(0)
    else:
        return x.sum(0) / x.shape[0]

class MeanPoolingFast(nn.Module):

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
        self.latent_embed_dim = cfg.latent_embed_dim if cfg.latent_embed_dim is not None else encoder_embed_dim
        logging.debug(f'| self.latent_embed_dim={self.latent_embed_dim!r}')
        self.linear = Linear(encoder_embed_dim, self.latent_embed_dim)
        self.projection = Linear(self.latent_embed_dim, num_targets)

    def forward(self, last_layer_feats, padding_mask, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Arguments\n            features - [TxBxD] Acoustic feature with shape\n            padding_mask - [BxT]     Padding Mask\n        '
        if padding_mask is not None:
            feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        else:
            feat_mask = None
        feat = self.linear(last_layer_feats)
        feat = fn_mean(feat, feat_mask)
        feat = self.activation_fn(feat)
        return self.projection(feat)

    def forward_latent(self, last_layer_feats, padding_mask, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Arguments\n            features - [TxBxD] Acoustic feature with shape\n            padding_mask - [BxT]     Padding Mask\n        '
        if padding_mask is not None:
            feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        else:
            feat_mask = None
        feat = self.linear(last_layer_feats)
        feat = fn_mean(feat, feat_mask)
        return feat

class MeanPoolingFastAMSoftmax(MeanPoolingFast):

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(cfg, encoder_embed_dim, num_targets, **kwargs)
        self.projection = Linear(self.latent_embed_dim, num_targets, bias=False)
        nn.init.xavier_normal_(self.projection.weight, gain=1)

    def forward(self, last_layer_feats, padding_mask, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arguments\n            features - [BxTxD] Acoustic feature with shape\n            padding_mask - [BxT]     Padding Mask\n        '
        feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        feat = self.linear(last_layer_feats)
        feat = fn_mean(feat, feat_mask)
        feat = self.activation_fn(feat)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        weight_norm = F.normalize(self.projection.weight.t(), p=2, dim=-1)
        cos_fw = feat_norm @ weight_norm
        return cos_fw

def fn_max(x, mask):
    if False:
        return 10
    '\n    Args:\n        x: TxBxD\n        mask: BxT\n    Return:\n        y: BxD\n    '
    mask = mask.t()[:, :, None].to(torch.bool)
    return x.masked_fill(~mask, -1e-08).max(0)[0]

class MaxPoolingFast(Pooling):

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int, **kwargs):
        if False:
            return 10
        super().__init__(cfg, encoder_embed_dim, num_targets)
        self.activation_fn = utils.get_activation_fn(cfg.activation_fn)
        self.linear = Linear(encoder_embed_dim, encoder_embed_dim)

    def forward(self, last_layer_feats, padding_mask, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Arguments\n            features - [TxBxD] Acoustic feature with shape\n            padding_mask - [BxT]     Padding Mask\n        '
        feat_mask = (~padding_mask).to(last_layer_feats.dtype)
        feat = self.linear(last_layer_feats)
        feat = fn_max(feat, feat_mask)
        feat = self.activation_fn(feat)
        return self.projection(feat)

class LayerWeightedMeanPooling(MeanPoolingFast):
    """Elmo-style weighted average representation."""

    def __init__(self, cfg: Wav2Vec2ClassificationConfig, encoder_embed_dim: int, num_targets: int, encoder_layers: int):
        if False:
            print('Hello World!')
        super().__init__(cfg, encoder_embed_dim, num_targets)
        self.num_layers = encoder_layers
        self.weights = nn.Parameter(torch.ones(encoder_layers))

    def forward(self, last_layer_feats, padding_mask, all_layer_feats):
        if False:
            while True:
                i = 10
        if not self.training:
            msg = f'Number of layers in input features = {len(all_layer_feats)}. Expected {self.num_layers} layers.'
            assert len(all_layer_feats) == self.num_layers, msg
        all_layer_feats_stacked = torch.stack(all_layer_feats, dim=0)
        (num_layers, *original_feat_shape) = all_layer_feats_stacked.shape
        all_layer_feats_stacked_flat = all_layer_feats_stacked.view(num_layers, -1)
        normalized_weights = F.softmax(self.weights, dim=-1)
        weighted_avg_features = (normalized_weights.unsqueeze(-1) * all_layer_feats_stacked_flat).sum(dim=0)
        weighted_avg_features = weighted_avg_features.view(*original_feat_shape)
        return super().forward(weighted_avg_features, padding_mask)