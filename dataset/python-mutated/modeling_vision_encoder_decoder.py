""" Classes to support Vision-Encoder-Text-Decoder architectures"""
import gc
import os
import tempfile
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    if False:
        while True:
            i = 10
    '\n    Shift input ids one token to the right.\n    '
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'VisionEncoderDecoderConfig'
VISION_ENCODER_DECODER_START_DOCSTRING = '\n    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model\n    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via\n    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]\n    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream\n    generative task, like image captioning.\n\n    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation\n    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation\n    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi\n    Zhou, Wei Li, Peter J. Liu.\n\n    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained\n    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical\n    character recognition (OCR) yields a significant performance improvement.\n\n    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any\n    other models (see the examples for more information).\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = "\n    Args:\n        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):\n            Pixel values. Pixel values can be obtained using an image processor (e.g. if you use ViT as the encoder,\n            you should use [`AutoImageProcessor`]). See [`ViTImageProcessor.__call__`] for details.\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n\n            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n\n            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the\n            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):\n            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor\n            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the\n            decoder.\n        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices\n            into associated vectors than the model's internal embedding lookup matrix.\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,\n            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored\n            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.\n        kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:\n\n            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.\n            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.\n"

@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class VisionEncoderDecoderModel(PreTrainedModel):
    """
    [`VisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base vision model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = 'vision_encoder_decoder'
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = True

    def __init__(self, config: Optional[PretrainedConfig]=None, encoder: Optional[PreTrainedModel]=None, decoder: Optional[PreTrainedModel]=None):
        if False:
            i = 10
            return i + 15
        if config is None and (encoder is None or decoder is None):
            raise ValueError('Either a configuration or an encoder and a decoder has to be provided.')
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        elif not isinstance(config, self.config_class):
            raise ValueError(f'Config: {config} has to be of type {self.config_class}')
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(f"If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for `config.encoder.hidden_size`.")
        config.tie_word_embeddings = False
        super().__init__(config)
        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)
        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(f'Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}')
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(f'Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}')
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(f'The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head')

    def get_encoder(self):
        if False:
            return 10
        return self.encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder

    def get_output_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        if False:
            print('Hello World!')
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Example:\n\n        ```python\n        >>> from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> image_processor = AutoImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")\n        >>> decoder_tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")\n        >>> model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> img = Image.open(requests.get(url, stream=True).raw)\n        >>> pixel_values = image_processor(images=img, return_tensors="pt").pixel_values  # Batch size 1\n\n        >>> output_ids = model.generate(\n        ...     pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True\n        ... ).sequences\n\n        >>> preds = decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)\n        >>> preds = [pred.strip() for pred in preds]\n\n        >>> assert preds == ["a cat laying on top of a couch next to another cat"]\n        ```'
        from_tf = kwargs.pop('from_tf', False)
        if from_tf:
            from transformers import TFVisionEncoderDecoderModel
            _tf_model = TFVisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            config = _tf_model.config
            encoder = _tf_model.encoder.__class__(_tf_model.config.encoder)
            decoder = _tf_model.decoder.__class__(_tf_model.config.decoder)
            encoder(encoder.dummy_inputs)
            decoder(decoder.dummy_inputs)
            encoder_variables = {}
            for v in encoder.trainable_variables + encoder.non_trainable_variables:
                encoder_variables['/'.join(v.name.split('/')[1:])] = v
            decoder_variables = {}
            for v in decoder.trainable_variables + decoder.non_trainable_variables:
                decoder_variables['/'.join(v.name.split('/')[1:])] = v
            _encoder_variables = {}
            for v in _tf_model.encoder.trainable_variables + _tf_model.encoder.non_trainable_variables:
                _encoder_variables['/'.join(v.name.split('/')[2:])] = v
            _decoder_variables = {}
            for v in _tf_model.decoder.trainable_variables + _tf_model.decoder.non_trainable_variables:
                _decoder_variables['/'.join(v.name.split('/')[2:])] = v
            for (name, v) in encoder_variables.items():
                v.assign(_encoder_variables[name])
            for (name, v) in decoder_variables.items():
                v.assign(_decoder_variables[name])
            tf_model = TFVisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
            if hasattr(_tf_model, 'enc_to_dec_proj'):
                tf_model(tf_model.dummy_inputs)
                tf_model.enc_to_dec_proj.kernel.assign(_tf_model.enc_to_dec_proj.kernel)
                tf_model.enc_to_dec_proj.bias.assign(_tf_model.enc_to_dec_proj.bias)
            with tempfile.TemporaryDirectory() as tmpdirname:
                encoder_dir = os.path.join(tmpdirname, 'encoder')
                decoder_dir = os.path.join(tmpdirname, 'decoder')
                tf_model.encoder.save_pretrained(encoder_dir)
                tf_model.decoder.save_pretrained(decoder_dir)
                if hasattr(tf_model, 'enc_to_dec_proj'):
                    enc_to_dec_proj_weight = torch.transpose(torch.from_numpy(tf_model.enc_to_dec_proj.kernel.numpy()), 1, 0)
                    enc_to_dec_proj_bias = torch.from_numpy(tf_model.enc_to_dec_proj.bias.numpy())
                del _tf_model
                del tf_model
                gc.collect()
                model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_dir, decoder_dir, encoder_from_tf=True, decoder_from_tf=True)
                model.config = config
                if hasattr(model, 'enc_to_dec_proj'):
                    model.enc_to_dec_proj.weight.data = enc_to_dec_proj_weight.contiguous()
                    model.enc_to_dec_proj.bias.data = enc_to_dec_proj_bias.contiguous()
                return model
        if kwargs.get('_fast_init', False):
            logger.warning('Fast initialization is currently not supported for VisionEncoderDecoderModel. Falling back to slow initialization...')
        kwargs['_fast_init'] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(cls, encoder_pretrained_model_name_or_path: str=None, decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
        if False:
            print('Hello World!')
        '\n        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model\n        checkpoints.\n\n\n        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train\n        the model, you need to first set it back in training mode with `model.train()`.\n\n        Params:\n            encoder_pretrained_model_name_or_path (`str`, *optional*):\n                Information necessary to initiate the image encoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An\n                      example is `google/vit-base-patch16-224-in21k`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n\n            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the text decoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaning positional arguments will be passed to the underlying model\'s `__init__` method.\n\n            kwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.\n                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import VisionEncoderDecoderModel\n\n        >>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized\n        >>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(\n        ...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./vit-bert")\n        >>> # load fine-tuned model\n        >>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")\n        ```'
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
                (encoder_config, kwargs_encoder) = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_encoder['config'] = encoder_config
            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        decoder = kwargs_decoder.pop('model', None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_decoder:
                (decoder_config, kwargs_decoder) = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                kwargs_decoder['config'] = decoder_config
            if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
                logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`')
            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoProcessor, VisionEncoderDecoderModel\n        >>> import requests\n        >>> from PIL import Image\n        >>> import torch\n\n        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")\n        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")\n\n        >>> # load image from the IAM dataset\n        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")\n\n        >>> # training\n        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id\n        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id\n        >>> model.config.vocab_size = model.config.decoder.vocab_size\n\n        >>> pixel_values = processor(image, return_tensors="pt").pixel_values\n        >>> text = "hello world"\n        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids\n        >>> outputs = model(pixel_values=pixel_values, labels=labels)\n        >>> loss = outputs.loss\n\n        >>> # inference (generation)\n        >>> generated_ids = model.generate(pixel_values)\n        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_encoder = {argument: value for (argument, value) in kwargs.items() if not argument.startswith('decoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('decoder_')}
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError('You have to specify pixel_values')
            encoder_outputs = self.encoder(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs_encoder)
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        encoder_hidden_states = encoder_outputs[0]
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        encoder_attention_mask = None
        if labels is not None and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, past_key_values=past_key_values, return_dict=return_dict, **kwargs_decoder)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        return Seq2SeqLMOutput(loss=loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            while True:
                i = 10
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            print('Hello World!')
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs['attention_mask'] if 'attention_mask' in decoder_inputs else None
        input_dict = {'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'decoder_input_ids': decoder_inputs['input_ids'], 'encoder_outputs': encoder_outputs, 'past_key_values': decoder_inputs['past_key_values'], 'use_cache': use_cache}
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError('Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))')

    def _reorder_cache(self, past_key_values, beam_idx):
        if False:
            print('Hello World!')
        return self.decoder._reorder_cache(past_key_values, beam_idx)