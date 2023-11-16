import itertools
import random
from collections import OrderedDict
from typing import NamedTuple, Optional, List, Tuple
import torch
from torch import nn, FloatTensor, IntTensor, Tensor
import torch.nn.functional as F
import torchvision
import torchvision.ops.boxes as box_ops
from allennlp.common import Registrable

class RegionDetectorOutput(NamedTuple):
    """
    The output type from the forward pass of a `RegionDetector`.
    """
    features: List[Tensor]
    '\n    A list of tensors, each with shape `(num_boxes, feature_dim)`.\n    '
    boxes: List[Tensor]
    '\n    A list of tensors containing the coordinates for each box. Each has shape `(num_boxes, 4)`.\n    '
    class_probs: Optional[List[Tensor]] = None
    '\n    An optional list of tensors. These tensors can have shape `(num_boxes,)` or\n    `(num_boxes, *)` if probabilities for multiple classes are given.\n    '
    class_labels: Optional[List[Tensor]] = None
    '\n    An optional list of tensors that give the labels corresponding to the `class_probs`\n    tensors. This should be non-`None` whenever `class_probs` is, and each tensor\n    should have the same shape as the corresponding tensor from `class_probs`.\n    '

class RegionDetector(nn.Module, Registrable):
    """
    A `RegionDetector` takes a batch of images, their sizes, and an ordered dictionary
    of image features as input, and finds regions of interest (or "boxes") within those images.

    Those regions of interest are described by three values:

    - `features` (`List[Tensor]`): A feature vector for each region, which is a tensor of shape
      `(num_boxes, feature_dim)`.
    - `boxes` (`List[Tensor]`): The coordinates of each region within the original image, with shape
      `(num_boxes, 4)`.
    - `class_probs` (`Optional[List[Tensor]]`): Class probabilities from some object
      detector that was used to find the regions of interest, with shape `(num_boxes,)`
      or `(num_boxes, *)` if probabilities for more than one class are given.
    - `class_labels` (`Optional[List[Tensor]]`): The labels corresponding to `class_probs`.
      Each tensor in this list has the same shape as the corresponding tensor in `class_probs`.

    """

    def forward(self, images: FloatTensor, sizes: IntTensor, image_features: 'OrderedDict[str, FloatTensor]') -> RegionDetectorOutput:
        if False:
            print('Hello World!')
        raise NotImplementedError()

@RegionDetector.register('random')
class RandomRegionDetector(RegionDetector):
    """
    A `RegionDetector` that returns two proposals per image, for testing purposes.  The features for
    the proposal are a random 10-dimensional vector, and the coordinates are the size of the image.
    """

    def __init__(self, seed: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.random = random.Random(seed)

    def _seeded_random_tensor(self, *shape: int, device) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        "PyTorch's random functions can't take a random seed. There is only one global\n        random seed in torch, but that's not deterministic enough for us. So we use Python's\n        random source to make random tensors."
        result = torch.zeros(*shape, dtype=torch.float32, device=device)
        for coordinates in itertools.product(*(range(size) for size in result.shape)):
            result[coordinates] = self.random.uniform(-1, 1)
        return result

    def forward(self, images: FloatTensor, sizes: IntTensor, image_features: 'OrderedDict[str, FloatTensor]') -> RegionDetectorOutput:
        if False:
            return 10
        (batch_size, num_features, height, width) = images.size()
        features = [self._seeded_random_tensor(2, 10, device=images.device) for _ in range(batch_size)]
        boxes = [torch.zeros(2, 4, dtype=torch.float32, device=images.device) for _ in range(batch_size)]
        for image_num in range(batch_size):
            boxes[image_num][0, 2] = sizes[image_num, 0]
            boxes[image_num][0, 3] = sizes[image_num, 1]
            boxes[image_num][1, 2] = sizes[image_num, 0]
            boxes[image_num][1, 3] = sizes[image_num, 1]
        return RegionDetectorOutput(features, boxes)

@RegionDetector.register('faster_rcnn')
class FasterRcnnRegionDetector(RegionDetector):
    """
    A [Faster R-CNN](https://arxiv.org/abs/1506.01497) pretrained region detector.

    Unless you really know what you're doing, this should be used with the image
    features created from the `ResnetBackbone` `GridEmbedder` and on images loaded
    using the `TorchImageLoader` with the default settings.


    !!! Note
        This module does not have any trainable parameters by default.
        All pretrained weights are frozen.

    # Parameters

    box_score_thresh : `float`, optional (default = `0.05`)
        During inference, only proposal boxes / regions with a label classification score
        greater than `box_score_thresh` will be returned.

    box_nms_thresh : `float`, optional (default = `0.5`)
        During inference, non-maximum suppression (NMS) will applied to groups of boxes
        that share a common label.

        NMS iteratively removes lower scoring boxes which have an intersection-over-union (IoU)
        greater than `box_nms_thresh` with another higher scoring box.

    max_boxes_per_image : `int`, optional (default = `100`)
        During inference, at most `max_boxes_per_image` boxes will be returned. The
        number of boxes returned will vary by image and will often be lower
        than `max_boxes_per_image` depending on the values of `box_score_thresh`
        and `box_nms_thresh`.
    """

    def __init__(self, *, box_score_thresh: float=0.05, box_nms_thresh: float=0.5, max_boxes_per_image: int=100):
        if False:
            while True:
                i = 10
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh, box_detections_per_img=max_boxes_per_image)
        del self.detector.backbone
        for parameter in self.detector.parameters():
            parameter.requires_grad = False

    def forward(self, images: FloatTensor, sizes: IntTensor, image_features: 'OrderedDict[str, FloatTensor]') -> RegionDetectorOutput:
        if False:
            while True:
                i = 10
        '\n        Extract regions and region features from the given images.\n\n        In most cases `image_features` should come directly from the `ResnetBackbone`\n        `GridEmbedder`. The `images` themselves should be standardized and resized\n        using the default settings for the `TorchImageLoader`.\n        '
        if self.training:
            raise RuntimeError('FasterRcnnRegionDetector can not be used for training at the moment')
        image_shapes: List[Tuple[int, int]] = list(((int(h), int(w)) for (h, w) in sizes))
        image_list = torchvision.models.detection.image_list.ImageList(images, image_shapes)
        proposals: List[Tensor]
        (proposals, _) = self.detector.rpn(image_list, image_features)
        box_features = self.detector.roi_heads.box_roi_pool(image_features, proposals, image_shapes)
        box_features = self.detector.roi_heads.box_head(box_features)
        (class_logits, box_regression) = self.detector.roi_heads.box_predictor(box_features)
        (boxes, features, scores, labels) = self._postprocess_detections(class_logits, box_features, box_regression, proposals, image_shapes)
        return RegionDetectorOutput(features, boxes, scores, labels)

    def _postprocess_detections(self, class_logits: Tensor, box_features: Tensor, box_regression: Tensor, proposals: List[Tensor], image_shapes: List[Tuple[int, int]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        if False:
            while True:
                i = 10
        '\n        Adapted from https://github.com/pytorch/vision/blob/\n        4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/models/detection/roi_heads.py#L664.\n\n        The only reason we have to re-implement this method is so we can pull out the box\n        features that we want.\n        '
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.detector.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        features_list = box_features.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_features = []
        all_scores = []
        all_labels = []
        for (boxes, features, scores, image_shape) in zip(pred_boxes_list, features_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            features = features.unsqueeze(1).expand(boxes.shape[0], boxes.shape[1], -1)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:]
            features = features[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 4)
            features = features.reshape(boxes.shape[0], -1)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.where(scores > self.detector.roi_heads.score_thresh)[0]
            (boxes, features, scores, labels) = (boxes[inds], features[inds], scores[inds], labels[inds])
            keep = box_ops.remove_small_boxes(boxes, min_size=0.01)
            (boxes, features, scores, labels) = (boxes[keep], features[keep], scores[keep], labels[keep])
            keep = box_ops.batched_nms(boxes, scores, labels, self.detector.roi_heads.nms_thresh)
            keep = keep[:self.detector.roi_heads.detections_per_img]
            (boxes, features, scores, labels) = (boxes[keep], features[keep], scores[keep], labels[keep])
            all_boxes.append(boxes)
            all_features.append(features)
            all_scores.append(scores)
            all_labels.append(labels)
        return (all_boxes, all_features, all_scores, all_labels)