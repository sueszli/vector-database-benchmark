""" OpenAI ImageGPT configuration"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, TensorType
logger = logging.get_logger(__name__)
IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {'openai/imagegpt-small': '', 'openai/imagegpt-medium': '', 'openai/imagegpt-large': ''}

class ImageGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ImageGPTModel`] or a [`TFImageGPTModel`]. It is
    used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
    [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ImageGPTModel`] or [`TFImageGPTModel`].
        n_positions (`int`, *optional*, defaults to 32*32):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 512):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # Initializing a ImageGPT configuration
    >>> configuration = ImageGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ImageGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'imagegpt'
    keys_to_ignore_at_inference = ['past_key_values']
    attribute_map = {'hidden_size': 'n_embd', 'max_position_embeddings': 'n_positions', 'num_attention_heads': 'n_head', 'num_hidden_layers': 'n_layer'}

    def __init__(self, vocab_size=512 + 1, n_positions=32 * 32, n_embd=512, n_layer=24, n_head=8, n_inner=None, activation_function='quick_gelu', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, scale_attn_weights=True, use_cache=True, tie_word_embeddings=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class ImageGPTOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if False:
            i = 10
            return i + 15
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'})])

    def generate_dummy_inputs(self, preprocessor: 'FeatureExtractionMixin', batch_size: int=1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None, num_channels: int=3, image_width: int=32, image_height: int=32) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        "\n        Generate inputs to provide to the ONNX exporter for the specific framework\n\n        Args:\n            preprocessor ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):\n                The preprocessor associated with this model configuration.\n            batch_size (`int`, *optional*, defaults to -1):\n                The batch size to export the model for (-1 means dynamic axis).\n            num_choices (`int`, *optional*, defaults to -1):\n                The number of candidate answers provided for multiple choice task (-1 means dynamic axis).\n            seq_length (`int`, *optional*, defaults to -1):\n                The sequence length to export the model for (-1 means dynamic axis).\n            is_pair (`bool`, *optional*, defaults to `False`):\n                Indicate if the input is a pair (sentence 1, sentence 2)\n            framework (`TensorType`, *optional*, defaults to `None`):\n                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.\n            num_channels (`int`, *optional*, defaults to 3):\n                The number of channels of the generated images.\n            image_width (`int`, *optional*, defaults to 40):\n                The width of the generated images.\n            image_height (`int`, *optional*, defaults to 40):\n                The height of the generated images.\n\n        Returns:\n            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function\n        "
        input_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
        inputs = dict(preprocessor(images=input_image, return_tensors=framework))
        return inputs