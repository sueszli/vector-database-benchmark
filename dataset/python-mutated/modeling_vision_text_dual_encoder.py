""" PyTorch VisionTextDualEncoder model."""
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from ..clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VisionTextDualEncoderConfig'
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = '\n    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model\n    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded\n    via the [`~AutoModel.from_pretrained`] method. The projection layers are automatically added to the model and\n    should be fine-tuned on a downstream task, like contrastive image-text modeling.\n\n    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how\n    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment\n    on new zero-shot vision tasks such as image classification or retrieval.\n\n    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other\n    models (see the examples for more information).\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = '\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = '\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using\n            an image processor (e.g. if you use ViT as the encoder, you should use [`AutoImageProcessor`]). See\n            [`ViTImageProcessor.__call__`] for details.\n        return_loss (`bool`, *optional*):\n            Whether or not to return the contrastive loss.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n'

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    if False:
        return 10
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class VisionTextDualEncoderModel(PreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = 'vision_text_dual_encoder'

    def __init__(self, config: Optional[VisionTextDualEncoderConfig]=None, vision_model: Optional[PreTrainedModel]=None, text_model: Optional[PreTrainedModel]=None):
        if False:
            print('Hello World!')
        if config is None and (vision_model is None or text_model is None):
            raise ValueError('Either a configuration or an vision and a text model has to be provided')
        if config is None:
            config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        elif not isinstance(config, self.config_class):
            raise ValueError(f'config: {config} has to be of type {self.config_class}')
        super().__init__(config)
        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)
        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config
        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids=None, attention_mask=None, position_ids=None, token_type_ids=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by\n            applying the projection layer to the pooled output of [`CLIPTextModel`].\n\n        Examples:\n\n        ```python\n        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer\n\n        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")\n        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")\n\n        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")\n        >>> text_features = model.get_text_features(**inputs)\n        ```'
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if False:
            return 10
        '\n        Returns:\n            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by\n            applying the projection layer to the pooled output of [`CLIPVisionModel`].\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import VisionTextDualEncoderModel, AutoImageProcessor\n\n        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")\n        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n\n        >>> inputs = image_processor(images=image, return_tensors="pt")\n\n        >>> image_features = model.get_image_features(**inputs)\n        ```'
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, token_type_ids: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        if False:
            return 10
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from PIL import Image\n        >>> import requests\n        >>> from transformers import (\n        ...     VisionTextDualEncoderModel,\n        ...     VisionTextDualEncoderProcessor,\n        ...     AutoImageProcessor,\n        ...     AutoTokenizer,\n        ... )\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")\n        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")\n        >>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)\n        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(\n        ...     "google/vit-base-patch16-224", "bert-base-uncased"\n        ... )\n\n        >>> # contrastive training\n        >>> urls = [\n        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",\n        ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",\n        ... ]\n        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]\n        >>> inputs = processor(\n        ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True\n        ... )\n        >>> outputs = model(\n        ...     input_ids=inputs.input_ids,\n        ...     attention_mask=inputs.attention_mask,\n        ...     pixel_values=inputs.pixel_values,\n        ...     return_loss=True,\n        ... )\n        >>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score\n\n        >>> # save and load from pretrained\n        >>> model.save_pretrained("vit-bert")\n        >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")\n\n        >>> # inference\n        >>> outputs = model(**inputs)\n        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score\n        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return CLIPOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['_fast_init'] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(cls, vision_model_name_or_path: str=None, text_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
        if False:
            i = 10
            return i + 15
        '\n        Params:\n            vision_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the vision model. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`\n                      should be set to `True` and a configuration object should be provided as `config` argument. This\n                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided\n                      conversion scripts and loading the Flax model afterwards.\n\n            text_model_name_or_path (`str`, *optional*):\n                Information necessary to initiate the text model. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`\n                      should be set to `True` and a configuration object should be provided as `config` argument. This\n                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided\n                      conversion scripts and loading the Flax model afterwards.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaning positional arguments will be passed to the underlying model\'s `__init__` method.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the text configuration, use the prefix *text_* for each configuration parameter.\n                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import VisionTextDualEncoderModel\n\n        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.\n        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(\n        ...     "google/vit-base-patch16-224", "bert-base-uncased"\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./vit-bert")\n        >>> # load fine-tuned model\n        >>> model = VisionTextDualEncoderModel.from_pretrained("./vit-bert")\n        ```'
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
                vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            else:
                kwargs_vision['config'] = vision_config
                vision_model = AutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
        text_model = kwargs_text.pop('model', None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError('If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined')
            if 'config' not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text['config'] = text_config
            text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config, **kwargs)
        model = cls(config=config, vision_model=vision_model, text_model=text_model)
        logger.warning("The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight', 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
        return model