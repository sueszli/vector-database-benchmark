from typing import Any, Dict
import torch
from PIL import Image
from torchvision import transforms
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor

class OfaVisualQuestionAnsweringPreprocessor(OfaBasePreprocessor):
    """
    OFA preprocessor for question answer tasks.
    """

    def __init__(self, cfg, model_dir, mode=ModeKeys.INFERENCE, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'preprocess the data\n\n        Args:\n            cfg(modelscope.utils.config.ConfigDict) : model config\n            model_dir (str): model path,\n            mode: preprocessor mode (model mode)\n        '
        super(OfaVisualQuestionAnsweringPreprocessor, self).__init__(cfg, model_dir, mode, *args, **kwargs)
        self.patch_resize_transform = transforms.Compose([lambda image: image.convert('RGB'), transforms.Resize((self.patch_image_size, self.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Building training samples.\n\n        step 1. Preprocess the data using the logic of `_build_infer_sample`\n            and make sure the label data in the result.\n        step 2. Preprocessing the label data to generate `target` and `prev_output_token`.\n            - add blank in the front out label data and tokenize it as `target` item.\n            - if `prompt_type` is `None`, add the bos token as previous output tokens,\n            add eos tokens as target items.\n            - if `prompt_type` is `src`, concatenate source text input with target item as\n            previous output tokens, remove the bos token and add eos token as target items.\n            - if `prompt_type` is `prev_output`, just like the `prompt_type` is src, the\n            difference is that it will remove the eos token in source text input in this\n            setting.\n            - padding the source item as final target item.\n        step 3. Add constraint mask.\n\n        Args:\n            data (`Dict[str, Any]`): Input data, should contains the key of `image`\n                `text` and `label`.\n        Return:\n            A dict object, contains source text input, patch images, patch masks\n            with `Tensor([True])`, decoder prompt, label, target previous output tokens\n            and constraint mask.\n        '
        sample = self._build_infer_sample(data)
        tgt_item = self.tokenize_text(' {}'.format(sample['label']), add_bos=False, add_eos=False)
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([sample['source'], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([sample['source'][:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item) - 1] = self.tokenizer.pad_token_id
        sample['prev_output_tokens'] = prev_output_item
        sample['target'] = target_item
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item) - len(tgt_item) - 1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            sample['constraint_mask'] = constraint_mask
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Building inference samples.\n\n        step 1. Preprocessing image input for model's image input.\n            - get pillow image from data.\n            - do some transforms to the pillow image, such as resize, normalize etc.\n        step 2. Preprocessing the text input for model's text input.\n            - add blank in the front of input text.\n            - tokenize the result above as source text input.\n        step 3. Calculating the decoder prompt.\n            - if `prompt_type` is `None`, using bos token.\n            - if `prompt_type` is `src`, using source text input\n            - if `prompt_type` is `prev_output`, using source text input without eos token.\n        step 4. Whether or not to add label data which refer to an answer to the question\n            in this task.\n\n        Args:\n            data (`Dict[str, Any]`): Input data, should contains the key of `image`\n                `text`.\n        Return:\n            A dict object, contains source text input, patch images, patch masks\n            with `Tensor([True])`, decoder prompt and label.\n        "
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        text = data[self.column_map['text']]
        text = self.pre_question(text, self.max_src_length)
        text = text + '?' if not text.endswith('?') else text
        inputs = self.tokenize_text(f' {text}')
        if self.prompt_type == 'none':
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            decoder_prompt = inputs
        elif self.prompt_type == 'prev_output':
            decoder_prompt = inputs[:-1]
        else:
            raise NotImplementedError
        sample = {'source': inputs, 'patch_image': patch_image, 'patch_mask': torch.tensor([True]), 'decoder_prompt': decoder_prompt}
        if 'answer' in self.column_map and self.column_map['answer'] in data:
            sample['label'] = data[self.column_map['answer']]
        return sample