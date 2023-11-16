""" Classes to support Vision-Encoder-Text-Decoder architectures"""
import os
from typing import Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VisionEncoderDecoderConfig'
VISION_ENCODER_DECODER_START_DOCSTRING = '\n    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model\n    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via\n    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]\n    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream\n    generative task, like image captioning.\n\n    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation\n    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation\n    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi\n    Zhou, Wei Li, Peter J. Liu.\n\n    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained\n    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical\n    character recognition (OCR) yields a significant performance improvement.\n\n    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any\n    other models (see the examples for more information).\n\n    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a Flax Linen\n    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a\n    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.\n\n    Parameters:\n        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using the vision model's image processor. For example, using\n            [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`] for details.\n        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the\n            range `[0, config.decoder.max_position_embeddings - 1]`.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            If set to `True`, the model will return a [`~utils.FlaxSeq2SeqLMOutput`] instead of a plain tuple.\n"
VISION_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using the vision model's image processor. For example, using\n            [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            If set to `True`, the model will return a [`~utils.FlaxBaseModelOutput`] instead of a plain tuple.\n"
VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING = '\n    Args:\n        decoder_input_ids (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n\n            For sequence to sequence training, `decoder_input_ids` should be provided. If no `decoder_input_ids` is\n            provided, the model will create this tensor by shifting the `input_ids` to the right for denoising\n            pre-training.\n        encoder_outputs (`tuple(tuple(jnp.ndarray)`):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of\n            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        decoder_attention_mask (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        decoder_position_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the\n            range `[0, config.decoder.max_position_embeddings - 1]`.\n        past_key_values (`Dict[str, jnp.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):\n            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast\n            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            If set to `True`, the model will return a [`~utils.FlaxCausalLMOutputWithCrossAttentions`] instead of a\n            plain tuple.\n'

class FlaxVisionEncoderDecoderModule(nn.Module):
    config: VisionEncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            i = 10
            return i + 15
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            self.enc_to_dec_proj = nn.Dense(self.decoder.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range), dtype=self.dtype)
        else:
            self.enc_to_dec_proj = None

    def _get_encoder_module(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def _get_projection_module(self):
        if False:
            return 10
        return self.enc_to_dec_proj

    def _get_decoder_module(self):
        if False:
            while True:
                i = 10
        return self.decoder

    def __call__(self, pixel_values, decoder_input_ids, decoder_attention_mask, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        if False:
            for i in range(10):
                print('nop')
        encoder_outputs = self.encoder(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        encoder_hidden_states = encoder_outputs[0]
        if self.enc_to_dec_proj is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        (batch_size, sequence_length) = encoder_hidden_states.shape[:2]
        encoder_attention_mask = jnp.ones((batch_size, sequence_length))
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return FlaxSeq2SeqLMOutput(logits=decoder_outputs.logits, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class FlaxVisionEncoderDecoderModel(FlaxPreTrainedModel):
    """
    [`FlaxVisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture
    with the module (flax.nn.Module) of one of the base vision model classes of the library as encoder module and
    another one as decoder module when created with the :meth*~transformers.FlaxAutoModel.from_pretrained* class method
    for the encoder and :meth*~transformers.FlaxAutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = 'vision_encoder_decoder'
    main_input_name = 'pixel_values'
    module_class = FlaxVisionEncoderDecoderModule

    def __init__(self, config: VisionEncoderDecoderConfig, input_shape: Optional[Tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            return 10
        if not _do_init:
            raise ValueError('`FlaxVisionEncoderDecoderModel` cannot be created without initializing, `_do_init` must be `True`.')
        if input_shape is None:
            num_channels = getattr(config.encoder, 'num_channels', 3)
            input_shape = ((1, config.encoder.image_size, config.encoder.image_size, num_channels), (1, 1))
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(f"If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for `config.encoder.hidden_size`.")
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            while True:
                i = 10
        (encoder_input_shape, decoder_input_shape) = input_shape
        pixel_values = jnp.zeros(encoder_input_shape, dtype=self.dtype)
        decoder_input_ids = jnp.zeros(decoder_input_shape, dtype='i4')
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        (batch_size, _, _, _) = pixel_values.shape
        (decoder_batch_size, decoder_sequence_length) = decoder_input_ids.shape
        if not decoder_batch_size == batch_size:
            raise ValueError(f'The inputs of encoder and decoder should have the same batch size, but got {batch_size} for encoder and {decoder_batch_size} for decoder.')
        decoder_position_ids = jnp.broadcast_to(jnp.arange(decoder_sequence_length)[None, :], (decoder_batch_size, decoder_sequence_length))
        (params_rng, dropout_rng) = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, pixel_values, decoder_input_ids, decoder_attention_mask, decoder_position_ids)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length, encoder_outputs):
        if False:
            print('Hello World!')
        '\n        Args:\n            batch_size (`int`):\n                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.\n            max_length (`int`):\n                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized\n                cache.\n            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):\n                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:\n                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)\n                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the\n                cross-attention of the decoder.\n        '
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            if False:
                while True:
                    i = 10
            decoder_module = module._get_decoder_module()
            return decoder_module(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, **kwargs)
        init_variables = self.module.init(jax.random.PRNGKey(0), decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, encoder_hidden_states=encoder_outputs[0], init_cache=True, method=_decoder_forward)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings(VISION_ENCODER_DECODER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def encode(self, pixel_values: jnp.ndarray, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, FlaxVisionEncoderDecoderModel\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")\n\n        >>> # initialize a vit-gpt2 from pretrained ViT and GPT2 models. Note that the cross-attention layers will be randomly initialized\n        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(\n        ...     "google/vit-base-patch16-224-in21k", "gpt2"\n        ... )\n\n        >>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values\n        >>> encoder_outputs = model.encode(pixel_values)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng

        def _encoder_forward(module, pixel_values, **kwargs):
            if False:
                i = 10
                return i + 15
            encode_module = module._get_encoder_module()
            return encode_module(pixel_values, **kwargs)
        outputs = self.module.apply({'params': params or self.params}, pixel_values=jnp.array(pixel_values, dtype=self.dtype), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, method=_encoder_forward)
        if return_dict:
            outputs = FlaxBaseModelOutput(last_hidden_state=outputs.last_hidden_state, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return outputs

    @add_start_docstrings(VISION_ENCODER_DECODER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def decode(self, decoder_input_ids, encoder_outputs, decoder_attention_mask: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, past_key_values: dict=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoImageProcessor, FlaxVisionEncoderDecoderModel\n        >>> import jax.numpy as jnp\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")\n\n        >>> # initialize a vit-gpt2 from pretrained ViT and GPT2 models. Note that the cross-attention layers will be randomly initialized\n        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(\n        ...     "google/vit-base-patch16-224-in21k", "gpt2"\n        ... )\n\n        >>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values\n        >>> encoder_outputs = model.encode(pixel_values)\n\n        >>> decoder_start_token_id = model.config.decoder.bos_token_id\n        >>> decoder_input_ids = jnp.ones((pixel_values.shape[0], 1), dtype="i4") * decoder_start_token_id\n\n        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)\n        >>> logits = outputs.logits\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        encoder_hidden_states = encoder_outputs[0]
        (batch_size, sequence_length) = encoder_hidden_states.shape[:2]
        encoder_attention_mask = jnp.ones((batch_size, sequence_length))
        (batch_size, sequence_length) = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `decoder_position_ids` when passing `past_key_values`.')
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, encoder_hidden_states, **kwargs):
            if False:
                return 10
            projection_module = module._get_projection_module()
            decoder_module = module._get_decoder_module()
            if projection_module is not None:
                encoder_hidden_states = projection_module(encoder_hidden_states)
            return decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, encoder_hidden_states, **kwargs)
        outputs = self.module.apply(inputs, decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=jnp.array(encoder_attention_mask, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, mutable=mutable, method=_decoder_forward)
        if past_key_values is not None and return_dict:
            (outputs, past) = outputs
            outputs['past_key_values'] = unfreeze(past['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            (outputs, past) = outputs
            outputs = outputs[:1] + (unfreeze(past['cache']),) + outputs[1:]
        return outputs

    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def __call__(self, pixel_values: jnp.ndarray, decoder_input_ids: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        if False:
            while True:
                i = 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import FlaxVisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")\n\n        >>> # load output tokenizer\n        >>> tokenizer_output = AutoTokenizer.from_pretrained("gpt2")\n\n        >>> # initialize a vit-gpt2 from pretrained ViT and GPT2 models. Note that the cross-attention layers will be randomly initialized\n        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(\n        ...     "google/vit-base-patch16-224-in21k", "gpt2"\n        ... )\n\n        >>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values\n\n        >>> # use GPT2\'s eos_token as the pad as well as eos token\n        >>> model.config.eos_token_id = model.config.decoder.eos_token_id\n        >>> model.config.pad_token_id = model.config.eos_token_id\n\n        >>> # generation\n        >>> sequences = model.generate(pixel_values, num_beams=4, max_length=12).sequences\n\n        >>> captions = tokenizer_output.batch_decode(sequences, skip_special_tokens=True)\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        if decoder_input_ids is None:
            raise ValueError("`decoder_input_ids` can't be `None`.")
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            (batch_size, sequence_length) = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        rngs = {'dropout': dropout_rng} if dropout_rng is not None else {}
        return self.module.apply({'params': params or self.params}, pixel_values=jnp.array(pixel_values, dtype=self.dtype), decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs)

    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, decoder_attention_mask: Optional[jax.Array]=None, encoder_outputs=None, **kwargs):
        if False:
            return 10
        (batch_size, seq_length) = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if decoder_attention_mask is not None:
            decoder_position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            decoder_position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype='i4')[None, :], (batch_size, seq_length))
        return {'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'decoder_attention_mask': extended_attention_mask, 'decoder_position_ids': decoder_position_ids}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        if False:
            for i in range(10):
                print('nop')
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['decoder_position_ids'] = model_kwargs['decoder_position_ids'][:, -1:] + 1
        return model_kwargs

    @classmethod
    def from_encoder_decoder_pretrained(cls, encoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]=None, decoder_pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]=None, *model_args, **kwargs) -> FlaxPreTrainedModel:
        if False:
            print('Hello World!')
        '\n        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model\n        checkpoints.\n\n        Params:\n            encoder_pretrained_model_name_or_path (`Union[str, os.PathLike]`, *optional*):\n                Information necessary to initiate the encoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An\n                      example is `google/vit-base-patch16-224-in21k`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n\n            decoder_pretrained_model_name_or_path (`Union[str, os.PathLike]`, *optional*, defaults to `None`):\n                Information necessary to initiate the decoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaning positional arguments will be passed to the underlying model\'s `__init__` method.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.\n                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import FlaxVisionEncoderDecoderModel\n\n        >>> # initialize a vit-gpt2 from a pretrained ViT and a pretrained GPT2 model. Note that the cross-attention layers will be randomly initialized\n        >>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(\n        ...     "google/vit-base-patch16-224-in21k", "gpt2"\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./vit-gpt2")\n        >>> # load fine-tuned model\n        >>> model = FlaxVisionEncoderDecoderModel.from_pretrained("./vit-gpt2")\n        ```'
        kwargs_encoder = {argument[len('encoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('decoder_')}
        for key in kwargs_encoder.keys():
            del kwargs['encoder_' + key]
        for key in kwargs_decoder.keys():
            del kwargs['decoder_' + key]
        encoder = kwargs_encoder.pop('model', None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_encoder:
                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_encoder['config'] = encoder_config
            encoder = FlaxAutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        decoder = kwargs_decoder.pop('model', None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_decoder:
                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                kwargs_decoder['config'] = decoder_config
            if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
                logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`')
            decoder = FlaxAutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        dtype = kwargs.pop('dtype', jnp.float32)
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        model = cls(config, dtype=dtype)
        model.params['encoder'] = encoder.params
        model.params['decoder'] = decoder.params
        return model