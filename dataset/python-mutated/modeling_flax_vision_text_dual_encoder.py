""" Flax VisionTextDualEncoder model."""
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_utils import FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import add_start_docstrings, logging
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FLAX_MODEL_MAPPING, FlaxAutoModel
from ..clip.modeling_flax_clip import FlaxCLIPOutput, FlaxCLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VisionTextDualEncoderConfig'
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = '\n    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model\n    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded\n    via the [`~FlaxAutoModel.from_pretrained`] method. The projection layers are automatically added to the model and\n    should be fine-tuned on a downstream task, like contrastive image-text modeling.\n\n    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how\n    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment\n    on new zero-shot vision tasks such as image classification or retrieval.\n\n    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other\n    models (see the examples for more information).\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n     This model is also a\n     [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it\n     as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and\n     behavior.\n\n    Finally, this model supports inherent JAX features such as:\n\n    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)\n    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)\n    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)\n    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)\n\n    Parameters:\n        config ([`VisionTextDualEncoderConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.\n        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):\n            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and\n            `jax.numpy.bfloat16` (on TPUs).\n\n            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If\n            specified all the computation will be performed with the given `dtype`.\n\n            **Note that this only specifies the dtype of the computation and does not influence the dtype of model\n            parameters.**\n\n            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and\n            [`~FlaxPreTrainedModel.to_bf16`].\n'
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            an image processor (e.g. if you use ViT as the encoder, you should use [`AutoImageProcessor`]). See\n            [`ViTImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

class FlaxVisionTextDualEncoderModule(nn.Module):
    config: VisionTextDualEncoderConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if False:
            print('Hello World!')
        vision_config = self.config.vision_config
        text_config = self.config.text_config
        self.vision_embed_dim = vision_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        self.projection_dim = self.config.projection_dim
        vision_module = FLAX_MODEL_MAPPING.get(self.config.vision_config.__class__, FlaxCLIPVisionModel).module_class
        text_module = FLAX_MODEL_MAPPING[self.config.text_config.__class__].module_class
        self.vision_model = vision_module(vision_config, dtype=self.dtype)
        self.text_model = text_module(text_config, dtype=self.dtype)
        self.visual_projection = nn.Dense(self.projection_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.02), use_bias=False)
        self.text_projection = nn.Dense(self.projection_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.02), use_bias=False)
        self.logit_scale = self.param('logit_scale', lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, [])

    def __call__(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, token_type_ids=None, deterministic: bool=True, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            print('Hello World!')
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        return FlaxCLIPOutput(logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class FlaxVisionTextDualEncoderModel(FlaxPreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    module_class = FlaxVisionTextDualEncoderModule

    def __init__(self, config: VisionTextDualEncoderConfig, input_shape: Optional[Tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if not _do_init:
            raise ValueError('`FlaxVisionTextDualEncoderModel` cannot be created without initializing, `_do_init` must be `True`.')
        if input_shape is None:
            input_shape = ((1, 1), (1, config.vision_config.image_size, config.vision_config.image_size, 3))
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        if False:
            for i in range(10):
                print('nop')
        input_ids = jnp.zeros(input_shape[0], dtype='i4')
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        token_type_ids = jnp.ones_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)
        pixel_values = jax.random.normal(rng, input_shape[1])
        (params_rng, dropout_rng) = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids, token_type_ids)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(self, input_ids, pixel_values, attention_mask=None, position_ids=None, token_type_ids=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        if False:
            print('Hello World!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        return self.module.apply({'params': params or self.params}, jnp.array(input_ids, dtype='i4'), jnp.array(pixel_values, dtype=jnp.float32), jnp.array(attention_mask, dtype='i4'), jnp.array(position_ids, dtype='i4'), jnp.array(token_type_ids, dtype='i4'), not train, output_attentions, output_hidden_states, return_dict, rngs=rngs)

    def get_text_features(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you\n                provide it.\n\n                Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n\n        Returns:\n            text_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The text embeddings obtained by applying\n            the projection layer to the pooled output of text model.\n        '
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, token_type_ids, deterministic):
            if False:
                print('Hello World!')
            text_outputs = module.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, deterministic=deterministic)
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return text_features
        return self.module.apply({'params': params or self.params}, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), jnp.array(position_ids, dtype='i4'), jnp.array(token_type_ids, dtype='i4'), not train, method=_get_features, rngs=rngs)

    def get_image_features(self, pixel_values, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train=False):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):\n                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained\n                using [`ImageFeatureExtractionMixin`]. See [`ImageFeatureExtractionMixin.__call__`] for details.\n\n        Returns:\n            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The image embeddings obtained by\n            applying the projection layer to the pooled output of vision model.\n        '
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            if False:
                while True:
                    i = 10
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            pooled_output = vision_outputs[1]
            image_features = module.visual_projection(pooled_output)
            return image_features
        return self.module.apply({'params': params or self.params}, jnp.array(pixel_values, dtype=jnp.float32), not train, method=_get_features, rngs=rngs)

    @classmethod
    def from_vision_text_pretrained(cls, vision_model_name_or_path: str=None, text_model_name_or_path: str=None, *model_args, **kwargs) -> FlaxPreTrainedModel:
        if False:
            i = 10
            return i + 15
        '\n        Params:\n            vision_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the vision model. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`\n                      should be set to `True` and a configuration object should be provided as `config` argument. This\n                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided\n                      conversion scripts and loading the Flax model afterwards.\n\n            text_model_name_or_path (`str`, *optional*):\n                Information necessary to initiate the text model. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`\n                      should be set to `True` and a configuration object should be provided as `config` argument. This\n                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided\n                      conversion scripts and loading the Flax model afterwards.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaning positional arguments will be passed to the underlying model\'s `__init__` method.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the text configuration, use the prefix *text_* for each configuration parameter.\n                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import FlaxVisionTextDualEncoderModel\n\n        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.\n        >>> model = FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(\n        ...     "google/vit-base-patch16-224", "bert-base-uncased"\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./vit-bert")\n        >>> # load fine-tuned model\n        >>> model = FlaxVisionTextDualEncoderModel.from_pretrained("./vit-bert")\n        ```'
        kwargs_vision = {argument[len('vision_'):]: value for (argument, value) in kwargs.items() if argument.startswith('vision_')}
        kwargs_text = {argument[len('text_'):]: value for (argument, value) in kwargs.items() if argument.startswith('text_')}
        for key in kwargs_vision.keys():
            del kwargs['vision_' + key]
        for key in kwargs_text.keys():
            del kwargs['text_' + key]
        vision_model = kwargs_vision.pop('model', None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError('If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined')
            if 'config' not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)
            if vision_config.model_type == 'clip':
                kwargs_vision['config'] = vision_config.vision_config
                vision_model = FlaxCLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            else:
                kwargs_vision['config'] = vision_config
                vision_model = FlaxAutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
        text_model = kwargs_text.pop('model', None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError('If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined')
            if 'config' not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text['config'] = text_config
            text_model = FlaxAutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)
        dtype = kwargs.pop('dtype', jnp.float32)
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config, **kwargs)
        model = cls(config, *model_args, dtype=dtype, **kwargs)
        model.params['vision_model'] = vision_model.params
        model.params['text_model'] = text_model.params
        logger.warning("The projection layer and logit scale weights `[('visual_projection', 'kernel'), ('text_projection', 'kernel'), ('logit_scale',)]` are newly initialized. You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
        return model
VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING = '\n    Returns:\n\n    Examples:\n\n    ```python\n    >>> from PIL import Image\n    >>> import requests\n    >>> import jax\n    >>> from transformers import (\n    ...     FlaxVisionTextDualEncoderModel,\n    ...     VisionTextDualEncoderProcessor,\n    ...     AutoImageProcessor,\n    ...     AutoTokenizer,\n    ... )\n\n    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n    >>> image_processor = AutoImageProcesor.from_pretrained("google/vit-base-patch16-224")\n    >>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)\n    >>> model = FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(\n    ...     "google/vit-base-patch16-224", "bert-base-uncased"\n    ... )\n\n    >>> # contrastive training\n    >>> urls = [\n    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",\n    ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",\n    ... ]\n    >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]\n    >>> inputs = processor(\n    ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="np", padding=True\n    ... )\n    >>> outputs = model(\n    ...     input_ids=inputs.input_ids,\n    ...     attention_mask=inputs.attention_mask,\n    ...     pixel_values=inputs.pixel_values,\n    ... )\n    >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n\n    >>> # save and load from pretrained\n    >>> model.save_pretrained("vit-bert")\n    >>> model = FlaxVisionTextDualEncoderModel.from_pretrained("vit-bert")\n\n    >>> # inference\n    >>> outputs = model(**inputs)\n    >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n    >>> probs = jax.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities\n    ```\n'
overwrite_call_docstring(FlaxVisionTextDualEncoderModel, VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING + VISION_TEXT_DUAL_ENCODER_MODEL_DOCSTRING)
append_replace_return_docstrings(FlaxVisionTextDualEncoderModel, output_type=FlaxCLIPOutput, config_class=_CONFIG_FOR_DOC)