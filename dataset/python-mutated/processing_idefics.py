"""
Processor class for IDEFICS.
"""
from typing import Callable, List, Optional, Union
from urllib.parse import urlparse
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available
if is_torch_available():
    import torch
IMAGE_TOKEN = '<image>'

def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    if False:
        return 10
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1
    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask

def image_attention_mask_for_packed_input_ids(input_ids, tokenizer):
    if False:
        return 10
    image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    next_image_attention_mask = torch.full_like(input_ids, fill_value=-1)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    eod_token_id = tokenizer.eos_token_id
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for (idx, token_id) in enumerate(input_ids[batch_idx]):
            if token_id == image_token_id:
                count += 1
                image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                image_attention_mask[batch_idx][idx] = count
            if seen_eod:
                image_attention_mask[batch_idx][idx] = -1
            if token_id == eod_token_id:
                seen_eod = True
    for batch_idx in range(input_ids.size(0)):
        count = -1
        seen_eod = False
        for idx in range(input_ids[batch_idx].size(0) - 1, -1, -1):
            token_id = input_ids[batch_idx][idx]
            if token_id == image_token_id:
                count += 1
                next_image_attention_mask[batch_idx][idx] = count
                seen_eod = False
            else:
                next_image_attention_mask[batch_idx][idx] = count
            if token_id == eod_token_id:
                seen_eod = True
            if seen_eod:
                next_image_attention_mask[batch_idx][idx] = -1
        non_negative_indices = next_image_attention_mask[batch_idx] != -1
        next_image_attention_mask[batch_idx][non_negative_indices] -= count
        next_image_attention_mask[batch_idx][non_negative_indices] *= -1
    return (image_attention_mask, next_image_attention_mask)

def is_url(string):
    if False:
        for i in range(10):
            print('nop')
    "Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately\n    invalidated the url"
    if ' ' in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])

class IdeficsProcessor(ProcessorMixin):
    """
    Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`IdeficsImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`IdeficsImageProcessor`):
            An instance of [`IdeficsImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_size (`int`, *optional*, defaults to 224): Image size (assuming a square image)
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'IdeficsImageProcessor'
    tokenizer_class = 'LlamaTokenizerFast'

    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        if False:
            print('Hello World!')
        if image_processor is None:
            raise ValueError('You need to specify an `image_processor`.')
        if tokenizer is None:
            raise ValueError('You need to specify a `tokenizer`.')
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.default_image_dims = (self.image_processor.image_num_channels, self.image_processor.image_size, self.image_processor.image_size)
        self.tokenizer_was_trained_with_end_of_utterance_token = True if '<end_of_utterance>' in self.tokenizer.special_tokens_map.get('additional_special_tokens', []) else False

    def __call__(self, prompts: Union[List[TextInput], List[List[TextInput]]], padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, transform: Callable=None, add_eos_token=False, add_end_of_utterance_token=None, debug=False, return_tensors: Optional[Union[str, TensorType]]=TensorType.PYTORCH) -> BatchEncoding:
        if False:
            i = 10
            return i + 15
        'This method takes batched or non-batched prompts made of text and images and converts them into prompts that\n        the model was trained on and prepares the image pixel values for the model to process.\n\n        Args:\n            prompts (`Union[List[TextInput], [List[List[TextInput]]]]`):\n                either a single prompt or a batched list of prompts - see the detailed description immediately after\n                the end of the arguments doc section.\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):\n                Select a strategy to pad the returned sequences (according to the model\'s padding side and padding\n                index) among:\n                - `True` or `\'longest\'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `\'max_length\'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `\'do_not_pad\'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            max_length (`int`, *optional*):\n                Maximum length of the returned list and optionally padding length (see above).\n            truncation (`bool`, *optional*):\n                Activates truncation to cut input sequences longer than `max_length` to `max_length`.\n            transform (`Callable`, *optional*):\n                A custom transform function that accepts a single image can be passed for training. For example,\n                `torchvision.Compose` can be used to compose multiple functions. If `None` a preset inference-specific\n                set of transforms will be applied to the images\n            add_eos_token (`bool`, *optional*, defaults to `False`):\n                Adds `eos_token` at the end of the final prompt if True`\n            add_end_of_utterance_token (`bool`, *optional*)\n                Whether to automatically add `<end_of_utterance>` after each prompt\'s text input (unless followed by an\n                image). If `None` the tokenizer will be checked instead and if this token is found in\n                `additional_special_tokens` then the value will be `True`.\n            debug (`bool`, *optional*, defaults to `False`):\n                `True` value will help debug prompt generation by dumping useful information\n            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`):\n                The type of tensors to return. Can be one of:\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n\n        Returns:\n            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be\n            directly passed to `model.generate`\n\n        Detailed explanation:\n\n        Each entry in `prompts` is either a text to be passed as is or an image that will be processed.\n\n        An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.\n\n        When the processor encounters an image it\'ll inject `<fake_token_around_image><image><fake_token_around_image>`\n        entry into the prompt.\n\n        Example:\n\n        ```python\n        checkpoint = "HuggingFaceM4/idefics-9b"\n        processor = AutoProcessor.from_pretrained(checkpoint)\n        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"\n        img = processor.image_processor.fetch_images([url])[0]\n\n        prompts = [\n            "User:",\n            img,\n            "Describe this image.\nAssistant: An image of two kittens in grass.\n",\n            "User:",\n            "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",\n            "Describe this image.\nAssistant:",\n        ]\n\n        inputs = processor(prompts, return_tensors="pt")\n        generated_ids = model.generate(**inputs, max_length=100)\n        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n        ```\n\n        In this example the `prompts` will be converted into:\n\n        ```\n        <s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.\n        Assistant: An image of two kittens in grass.\n        User:<fake_token_around_image><image><fake_token_around_image>Describe this image.\n        Assistant:\'\n        ```\n\n        and the two images will be massaged using [`IdeficsImageProcessor.__call__`] method and placed inside the\n        `pixel_values` dict entry of the return value.\n\n        This example also examplifies that images can be passed as objects or as text urls. It can be seen that the\n        first image is passed as object and the second one as a url.\n\n        To do training do:\n\n        ```python\n        image_transform = transforms.Compose(\n            [\n                transforms.RandomResizedCrop(\n                    (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC\n                ),\n                transforms.ToTensor(),\n                transforms.Normalize(mean=self.image_mean, std=self.image_std),\n            ]\n        )\n        inputs = processor(prompts, transform=image_transform, return_tensors="pt")\n        ```\n\n        In order to help debug prompt generation enable `debug=True` which will show you what\'s happening.\n\n        '
        if add_end_of_utterance_token is None:
            add_end_of_utterance_token = self.tokenizer_was_trained_with_end_of_utterance_token
        if not any((isinstance(i, list) for i in prompts)):
            prompts = [prompts]
        fake_token = '<fake_token_around_image>'
        image_token = '<image>'
        end_of_utterance_token = '<end_of_utterance>'

        def image_tokens(last_was_image):
            if False:
                return 10
            if last_was_image:
                return image_token + fake_token
            else:
                return fake_token + image_token + fake_token
        all_prompts = []
        all_images = []
        for sample in prompts:
            full_text = f'{self.tokenizer.bos_token}'
            image_objects = []
            last_was_image = False
            last_was_text = False
            for (i, item) in enumerate(sample):
                if i > 0:
                    last_was_text = True if not last_was_image else False
                if isinstance(item, str):
                    item = item.strip(' ')
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        image_objects.append(image)
                        last_was_image = True
                    else:
                        if add_end_of_utterance_token and last_was_text:
                            full_text += end_of_utterance_token
                        full_text += item
                        last_was_image = False
                else:
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item)
                    last_was_image = True
            if add_eos_token:
                full_text += self.tokenizer.eos_token
            if debug is True:
                print(f'full_text={full_text!r}')
            image_objects = self.image_processor(image_objects, transform=transform)
            all_prompts.append(full_text)
            all_images.append(image_objects)
        text_encoding = self.tokenizer(text=all_prompts, add_special_tokens=False, padding=padding, truncation=truncation, max_length=max_length)
        all_texts = text_encoding['input_ids']
        max_seq_len = max((len(x) for x in all_texts))
        max_num_images = max((len(x) for x in all_images))
        max_num_images = max(1, max_num_images)
        at_least_one_image = sum((len(x) for x in all_images)) > 0
        output_input_ids = []
        output_images = []
        output_attention_masks = []
        for (text, images) in zip(all_texts, all_images):
            padded_input_ids = [self.tokenizer.pad_token_id] * max_seq_len
            unpadded_seq_len = len(text)
            start = max_seq_len - unpadded_seq_len
            padded_input_ids[start:] = text[:max_seq_len]
            attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
            attention_mask[start:] = 1
            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)
            current_images = images[:local_max_num_images]
            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(max_num_images, *current_images.size()[1:])
                padded_image_tensor[:current_images.size(0)] = current_images
            else:
                padded_image_tensor = torch.zeros(max_num_images, *self.default_image_dims)
            output_images.append(padded_image_tensor)
            output_input_ids.append(torch.tensor(padded_input_ids))
            output_attention_masks.append(attention_mask)
        output_input_ids = torch.stack(output_input_ids)
        output_images = torch.stack(output_images)
        output_attention_masks = torch.stack(output_attention_masks)
        if at_least_one_image:
            (image_attention_mask, _) = image_attention_mask_for_packed_input_ids(output_input_ids, self.tokenizer)
            image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
        else:
            image_attention_mask = torch.zeros(output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool)
        return BatchFeature(data={'input_ids': output_input_ids, 'attention_mask': output_attention_masks, 'pixel_values': output_images, 'image_attention_mask': image_attention_mask})

    def batch_decode(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please\n        refer to the docstring of this method for more information.\n        "
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if False:
            return 10
        "\n        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to\n        the docstring of this method for more information.\n        "
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if False:
            i = 10
            return i + 15
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))