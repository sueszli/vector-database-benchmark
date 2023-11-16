""" Open-Llama model configuration"""
from ....configuration_utils import PretrainedConfig
from ....utils import logging
logger = logging.get_logger(__name__)
OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {'s-JoL/Open-Llama-V1': 'https://huggingface.co/s-JoL/Open-Llama-V1/blob/main/config.json'}

class OpenLlamaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`OpenLlamaModel`]. It is used to instantiate an
    Open-Llama model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Open-Llama model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`OpenLlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.

        Example:

    ```python
    >>> from transformers import OpenLlamaModel, OpenLlamaConfig

    >>> # Initializing a Open-Llama open_llama-7b style configuration
    >>> configuration = OpenLlamaConfig()

    >>> # Initializing a model from the open_llama-7b style configuration
    >>> model = OpenLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'open-llama'

    def __init__(self, vocab_size=100000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, hidden_act='silu', max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, use_memory_efficient_attention=True, hidden_dropout_prob=0.1, attention_dropout_prob=0.1, use_stable_embedding=True, shared_input_output_embedding=True, rope_scaling=None, **kwargs):
        if False:
            return 10
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_memory_efficient_attention = kwargs.pop('use_memorry_efficient_attention', use_memory_efficient_attention)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.use_stable_embedding = use_stable_embedding
        self.shared_input_output_embedding = shared_input_output_embedding
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    def _rope_scaling_validation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate the `rope_scaling` configuration.\n        '
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(f'`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, got {self.rope_scaling}')
        rope_scaling_type = self.rope_scaling.get('type', None)
        rope_scaling_factor = self.rope_scaling.get('factor', None)
        if rope_scaling_type is None or rope_scaling_type not in ['linear', 'dynamic']:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}")
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")