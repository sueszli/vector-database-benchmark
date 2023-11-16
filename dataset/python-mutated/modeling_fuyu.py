""" PyTorch Fuyu model."""
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fuyu import FuyuConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'FuyuConfig'
FUYU_START_DOCSTRING = '\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`FuyuConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'

@add_start_docstrings('The bare Fuyu Model outputting raw hidden-states without any specific head on top.', FUYU_START_DOCSTRING)
class FuyuPreTrainedModel(PreTrainedModel):
    config_class = FuyuConfig
    base_model_prefix = 'fuyu'
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
FUYU_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide\n            it.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see\n            `past_key_values`).\n\n            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]\n            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more\n            information on the default strategy.\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n        image_patches (`torch.FloatTensor` of shape `(batch_size, num_total_patches, patch_size_ x patch_size x num_channels)`, *optional*):\n            Image patches to be used as continuous embeddings. The patches are flattened and then projected to the\n            hidden size of the model.\n        image_patches_indices (`torch.LongTensor` of shape `(batch_size, num_total_patches + number_of_newline_tokens + number_of_text_tokens, patch_size_ x patch_size x num_channels )`, *optional*):\n            Indices indicating at which position the image_patches have to be inserted in input_embeds.\n        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.n_positions - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):\n            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape\n            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape\n            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.\n\n            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention\n            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.\n\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        use_cache (`bool`, *optional*):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.', FUYU_START_DOCSTRING)
class FuyuForCausalLM(FuyuPreTrainedModel):

    def __init__(self, config: FuyuConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.vision_embed_tokens = nn.Linear(config.patch_size * config.patch_size * config.num_channels, config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        if False:
            for i in range(10):
                print('nop')
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(self, word_embeddings: torch.Tensor, continuous_embeddings: List[torch.Tensor], image_patch_input_indices: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        'This function places the continuous_embeddings into the word_embeddings at the locations\n        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous\n        embeddings.\n\n        Args:\n            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):\n                Tensor of word embeddings.\n            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):\n                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape\n                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative\n                indices in image_patch_input_indices for that batch element.\n            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Tensor of indices of the image patches in the input_ids tensor.\n        '
        if not word_embeddings.shape[0] == len(continuous_embeddings):
            raise ValueError(f'Batch sizes must match! Got len(continuous_embeddings)={len(continuous_embeddings)!r} and word_embeddings.shape[0]={word_embeddings.shape[0]!r}')
        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(f'Number of continuous embeddings continuous_embeddings[batch_idx].shape={continuous_embeddings[batch_idx].shape!r} does not match number of continuous token ids src_indices.shape={src_indices.shape!r} in batch element {batch_idx}.')
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, image_patches: torch.Tensor=None, image_patches_indices: torch.Tensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CausalLMOutputWithPast]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import FuyuProcessor, FuyuForCausalLM\n        >>> from PIL import Image\n        >>> import requests\n\n        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")\n        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")\n\n        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n        >>> image = Image.open(requests.get(url, stream=True).raw)\n        >>> prompt = "Generate a coco-style caption.\\n"\n\n        >>> inputs = processor(text=text_prompt, images=image, return_tensors="pt")\n        >>> outputs = model(**inputs)\n\n        >>> generated_ids = model.generate(**model_inputs, max_new_tokens=7)\n        >>> generation_text = processor.batch_decode(generated_ids, skip_special_tokens=True)\n        >>> print(generation_text)\n        \'A bus parked on the side of a road.\'\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            (batch_size, seq_length) = input_ids.shape
        elif inputs_embeds is not None:
            (batch_size, seq_length, _) = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either input_is or inputs_embeds')
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0) for patch in image_patches]
                inputs_embeds = self.gather_continuous_embeddings(word_embeddings=inputs_embeds, continuous_embeddings=patch_embeddings, image_patch_input_indices=image_patches_indices)
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, labels=labels, use_cache=use_cache, return_dict=return_dict)
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, image_patches=None, image_patches_indices=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        if image_patches_indices is not None:
            model_inputs['image_patches_indices'] = image_patches_indices
        model_inputs.update({'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask, 'image_patches_indices': image_patches_indices if past_key_values is None else None, 'image_patches': image_patches if past_key_values is None else None})
        return model_inputs