""" MaskFormer model configuration"""
from typing import Dict, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..detr import DetrConfig
from ..swin import SwinConfig
MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {'facebook/maskformer-swin-base-ade': 'https://huggingface.co/facebook/maskformer-swin-base-ade/blob/main/config.json'}
logger = logging.get_logger(__name__)

class MaskFormerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MaskFormerModel`]. It is used to instantiate a
    MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MaskFormer
    [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) architecture trained
    on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, MaskFormer only supports the [Swin Transformer](swin) as backbone.

    Args:
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight to apply to the null (no object) class.
        use_auxiliary_loss(`bool`, *optional*, defaults to `False`):
            If `True` [`MaskFormerForInstanceSegmentationOutput`] will contain the auxiliary losses computed using the
            logits from each decoder's stage.
        backbone_config (`Dict`, *optional*):
            The configuration passed to the backbone, if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        decoder_config (`Dict`, *optional*):
            The configuration passed to the transformer decoder model, if unset the base config for `detr-resnet-50`
            will be used.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        dice_weight (`float`, *optional*, defaults to 1.0):
            The weight for the dice loss.
        cross_entropy_weight (`float`, *optional*, defaults to 1.0):
            The weight for the cross entropy loss.
        mask_weight (`float`, *optional*, defaults to 20.0):
            The weight for the mask loss.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.

    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]` or the decoder model type selected is not
            in `["detr"]`

    Examples:

    ```python
    >>> from transformers import MaskFormerConfig, MaskFormerModel

    >>> # Initializing a MaskFormer facebook/maskformer-swin-base-ade configuration
    >>> configuration = MaskFormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/maskformer-swin-base-ade style configuration
    >>> model = MaskFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    model_type = 'maskformer'
    attribute_map = {'hidden_size': 'mask_feature_size'}
    backbones_supported = ['resnet', 'swin']
    decoders_supported = ['detr']

    def __init__(self, fpn_feature_size: int=256, mask_feature_size: int=256, no_object_weight: float=0.1, use_auxiliary_loss: bool=False, backbone_config: Optional[Dict]=None, decoder_config: Optional[Dict]=None, init_std: float=0.02, init_xavier_std: float=1.0, dice_weight: float=1.0, cross_entropy_weight: float=1.0, mask_weight: float=20.0, output_auxiliary_logits: Optional[bool]=None, **kwargs):
        if False:
            print('Hello World!')
        if backbone_config is None:
            backbone_config = SwinConfig(image_size=384, in_channels=3, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12, drop_path_rate=0.3, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        if isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop('model_type')
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)
        if backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with MaskFormer. Supported model types: {','.join(self.backbones_supported)}")
        if decoder_config is None:
            decoder_config = DetrConfig()
        else:
            decoder_type = decoder_config.pop('model_type') if isinstance(decoder_config, dict) else decoder_config.model_type
            if decoder_type not in self.decoders_supported:
                raise ValueError(f"Transformer Decoder {decoder_type} not supported, please use one of {','.join(self.decoders_supported)}")
            if isinstance(decoder_config, dict):
                config_class = CONFIG_MAPPING[decoder_type]
                decoder_config = config_class.from_dict(decoder_config)
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config
        self.fpn_feature_size = fpn_feature_size
        self.mask_feature_size = mask_feature_size
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.use_auxiliary_loss = use_auxiliary_loss
        self.no_object_weight = no_object_weight
        self.output_auxiliary_logits = output_auxiliary_logits
        self.num_attention_heads = self.decoder_config.encoder_attention_heads
        self.num_hidden_layers = self.decoder_config.num_hidden_layers
        super().__init__(**kwargs)

    @classmethod
    def from_backbone_and_decoder_configs(cls, backbone_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs):
        if False:
            print('Hello World!')
        'Instantiate a [`MaskFormerConfig`] (or a derived class) from a pre-trained backbone model configuration and DETR model\n        configuration.\n\n            Args:\n                backbone_config ([`PretrainedConfig`]):\n                    The backbone configuration.\n                decoder_config ([`PretrainedConfig`]):\n                    The transformer decoder configuration to use.\n\n            Returns:\n                [`MaskFormerConfig`]: An instance of a configuration object\n        '
        return cls(backbone_config=backbone_config, decoder_config=decoder_config, **kwargs)