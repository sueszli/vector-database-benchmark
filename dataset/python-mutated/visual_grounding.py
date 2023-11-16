from typing import Any, Dict
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor
from .utils import transforms as T

class OfaVisualGroundingPreprocessor(OfaBasePreprocessor):
    """
    OFA preprocessor for visual grounding tasks.
    """

    def __init__(self, cfg, model_dir, mode=ModeKeys.INFERENCE, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'preprocess the data\n\n        Args:\n            cfg(modelscope.utils.config.ConfigDict) : model config\n            model_dir (str): model path,\n            mode: preprocessor mode (model mode)\n        '
        super(OfaVisualGroundingPreprocessor, self).__init__(cfg, model_dir, mode, *args, **kwargs)
        self.num_bins = self.cfg.model.get('num_bins', 1000)
        if self.mode == ModeKeys.TRAIN:
            self.positioning_transform = T.Compose([T.RandomResize([self.patch_image_size], max_size=self.patch_image_size), T.ToTensor(), T.Normalize(mean=self.mean, std=self.std, max_image_size=self.max_image_size)])
        else:
            self.patch_resize_transform = transforms.Compose([lambda image: image.convert('RGB'), transforms.Resize((self.patch_image_size, self.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Building training samples.\n\n        step 1. Preprocessing the image input for model\'s image input.\n            - get the pillow image.\n            - calculate the target boxes using for getting the exact area\n            in the pillow image for input text by input `region_coord`. in\n            training setting, `region_coord` will be a label data.\n            - getting the target image as patch images and do some transforms\n            such as resize, normalize etc.\n        step 2. Preprocessing the text input for model\'s source text input.\n            - do the str preprocessing to text input by function `pre_caption`.\n            - build the instruction. the default instruction is\n            ` which region does the text " {} " describe?`, `{}` refer to the\n            text input.\n            - tokenize the instruction as source text input.\n        step 3. Preprocessing the patch image boxes for model\'s target text input.\n            - quantize the coordinate of selected patch images\n            - concatenate the quantization results by blank\n            - tokenize the result above as target text input.\n        step 4. Get the previous output tokens using target item without eos token.\n\n        Args:\n            data (`Dict[str, Any]`): Input data, should contains the key of `image`\n                `text` and `region_coord`.\n        Return:\n            A dict object, contains source text input, patch images, patch masks\n            with `Tensor([True])` value, target, previous output tokens,\n            width scale ratio, height scale ratio and region coordinate.\n        '
        image = self.get_img_pil(data[self.column_map['image']])
        (w, h) = image.size
        boxes_target = {'boxes': [], 'labels': [], 'area': [], 'size': torch.tensor([h, w])}
        (x0, y0, x1, y1) = data[self.column_map['region_coord']].strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target['boxes'] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target['labels'] = np.array([0])
        area = [(float(x1) - float(x0)) * (float(y1) - float(y0))]
        boxes_target['area'] = torch.tensor(area)
        (patch_image, patch_boxes) = self.positioning_transform(image, boxes_target)
        (resize_h, resize_w) = (patch_boxes['size'][0], patch_boxes['size'][1])
        quant_x0 = '<bin_{}>'.format(int((patch_boxes['boxes'][0][0] * (self.num_bins - 1)).round()))
        quant_y0 = '<bin_{}>'.format(int((patch_boxes['boxes'][0][1] * (self.num_bins - 1)).round()))
        quant_x1 = '<bin_{}>'.format(int((patch_boxes['boxes'][0][2] * (self.num_bins - 1)).round()))
        quant_y1 = '<bin_{}>'.format(int((patch_boxes['boxes'][0][3] * (self.num_bins - 1)).round()))
        region_coord = '{} {} {} {}'.format(quant_x0, quant_y0, quant_x1, quant_y1)
        src_caption = self.pre_caption(data[self.column_map['text']], self.max_src_length)
        prompt = self.cfg.model.get('prompt', ' which region does the text " {} " describe?')
        text = prompt.format(src_caption)
        src_item = self.tokenize_text(text)
        target_item = self.tokenize_text(region_coord, add_bos=False)
        prev_output_item = torch.cat([self.bos_item, target_item[:-1]])
        sample = {'source': src_item, 'patch_image': patch_image, 'patch_mask': torch.tensor([True]), 'target': target_item, 'prev_output_tokens': prev_output_item, 'w_resize_ratio': resize_w / w, 'h_resize_ratio': resize_h / h, 'region_coord': region}
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Building inference samples.\n\n        step 1. Preprocessing image input for model\'s image input.\n            - get pillow image from data.\n            - do some transforms to the pillow image, such as resize, normalize etc.\n        step 2. Preprocessing the text input for model\'s text input.\n            - do the str preprocessing to text input by function `pre_caption`.\n            - build the instruction. the default instruction is\n            ` which region does the text " {} " describe?`, `{}` refer to the\n            text input.\n            - tokenize the instruction as source text input.\n        step 3. Whether or not to add label data which refer to a region coordinate\n            in this task.\n\n        Args:\n            data (`Dict[str, Any]`): Input data, should contains the key of `image`\n                `text`.\n        Return:\n            A dict object, contains source text input, patch images, patch masks\n            with `Tensor([True])` value, width scale ratio, height scale ratio\n            and label.\n        '
        image = self.get_img_pil(data[self.column_map['image']])
        (w, h) = image.size
        patch_image = self.patch_resize_transform(image)
        w_resize_ratio = torch.tensor(self.patch_image_size / w)
        h_resize_ratio = torch.tensor(self.patch_image_size / h)
        src_caption = self.pre_caption(data[self.column_map['text']], self.max_src_length)
        prompt = self.cfg.model.get('prompt', ' which region does the text " {} " describe?')
        text = prompt.format(src_caption)
        src_item = self.tokenize_text(text)
        sample = {'source': src_item, 'patch_image': patch_image, 'patch_mask': torch.tensor([True]), 'w_resize_ratio': w_resize_ratio, 'h_resize_ratio': h_resize_ratio}
        if 'region_coord' in self.column_map and self.column_map['region_coord'] in data:
            (x0, y0, x1, y1) = data[self.column_map['region_coord']].strip().split(',')
            sample['label'] = [float(x0), float(y0), float(x1), float(y1)]
        return sample