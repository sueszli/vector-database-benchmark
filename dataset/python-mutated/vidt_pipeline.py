from typing import Any, Dict
import torch
import torchvision.transforms as transforms
from torch import nn
from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.image_object_detection, module_name=Pipelines.vidt)
class VidtPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        use `model` to create a vidt pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        Example:\n            >>> from modelscope.pipelines import pipeline\n            >>> vidt_pipeline = pipeline('image-object-detection', 'damo/ViDT-logo-detection')\n            >>> result = vidt_pipeline(\n                'data/test/images/vidt_test1.png')\n            >>> print(f'Output: {result}.')\n        "
        super().__init__(model=model, **kwargs)
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize([640, 640]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.postprocessors = PostProcess()
        self.label_dic = {0: 'negative', 1: 'positive'}

    def preprocess(self, inputs: Input, **preprocess_params):
        if False:
            for i in range(10):
                print('nop')
        img = LoadImage.convert_to_img(inputs)
        ori_size = [img.size[1], img.size[0]]
        image = self.transform(img)
        tensor_list = [image]
        orig_target_sizes = [ori_size]
        orig_target_sizes = torch.tensor(orig_target_sizes).to(self.device)
        samples = nested_tensor_from_tensor_list(tensor_list)
        samples = samples.to(self.device)
        res = {}
        res['tensors'] = samples.tensors
        res['mask'] = samples.mask
        res['orig_target_sizes'] = orig_target_sizes
        return res

    def forward(self, inputs: Dict[str, Any], **forward_params):
        if False:
            return 10
        tensors = inputs['tensors']
        mask = inputs['mask']
        orig_target_sizes = inputs['orig_target_sizes']
        with torch.no_grad():
            (out_pred_logits, out_pred_boxes) = self.model(tensors, mask)
            res = {}
            res['out_pred_logits'] = out_pred_logits
            res['out_pred_boxes'] = out_pred_boxes
            res['orig_target_sizes'] = orig_target_sizes
            return res

    def postprocess(self, inputs: Dict[str, Any], **post_params):
        if False:
            for i in range(10):
                print('nop')
        results = self.postprocessors(inputs['out_pred_logits'], inputs['out_pred_boxes'], inputs['orig_target_sizes'])
        batch_predictions = get_predictions(results)[0]
        scores = []
        labels = []
        boxes = []
        for sub_pre in batch_predictions:
            scores.append(sub_pre[0])
            labels.append(self.label_dic[sub_pre[1]])
            boxes.append(sub_pre[2])
        outputs = {}
        outputs['scores'] = scores
        outputs['labels'] = labels
        outputs['boxes'] = boxes
        return outputs

def nested_tensor_from_tensor_list(tensor_list):
    if False:
        print('Hello World!')
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    (b, c, h, w) = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for (img, pad_img, m) in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[1], :img.shape[2]] = False
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    if False:
        print('Hello World!')
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for (index, item) in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):

    def __init__(self, tensors, mask):
        if False:
            return 10
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        if False:
            for i in range(10):
                print('nop')
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        if False:
            while True:
                i = 10
        return (self.tensors, self.mask)

    def __repr__(self):
        if False:
            return 10
        return str(self.tensors)

def box_cxcywh_to_xyxy(x):
    if False:
        for i in range(10):
            print('nop')
    (x_c, y_c, w, h) = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)

def get_predictions(post_results, bbox_thu=0.4):
    if False:
        for i in range(10):
            print('nop')
    batch_final_res = []
    for per_img_res in post_results:
        per_img_final_res = []
        for i in range(len(per_img_res['scores'])):
            score = float(per_img_res['scores'][i].cpu())
            label = int(per_img_res['labels'][i].cpu())
            bbox = []
            for it in per_img_res['boxes'][i].cpu():
                bbox.append(int(it))
            if score >= bbox_thu:
                per_img_final_res.append([score, label, bbox])
        batch_final_res.append(per_img_final_res)
    return batch_final_res

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, processor_dct=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, out_logits, out_bbox, target_sizes):
        if False:
            print('Hello World!')
        ' Perform the computation\n\n        Parameters:\n            out_logits: raw logits outputs of the model\n            out_bbox: raw bbox outputs of the model\n            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch\n                          For evaluation, this must be the original image size (before any data augmentation)\n                          For visualization, this should be the image size after data augment, but before padding\n        '
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        (topk_values, topk_indexes) = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        (img_h, img_w) = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.float32)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for (s, l, b) in zip(scores, labels, boxes)]
        return results