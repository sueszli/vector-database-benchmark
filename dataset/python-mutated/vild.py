import os
from typing import Any, Dict, Union
import clip
import numpy as np
import tensorflow.compat.v1 as tf
import torch.cuda
from scipy.special import softmax
from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_model import Model
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@MODELS.register_module(Tasks.open_vocabulary_detection, module_name=Models.open_vocabulary_detection_vild)
class OpenVocabularyDetectionViLD(Model):
    """
    Vild: Open-Vocabulary Detection via Vision and Language Knowledge Distillation
    https://arxiv.org/abs/2104.13921
    """

    def __init__(self, model_dir, *args, **kwargs):
        if False:
            return 10
        self.model_dir = model_dir
        device_name = kwargs.get('device', 'gpu')
        self._device_name = device_name
        model_path = os.path.join(model_dir, ModelFile.TF_GRAPH_FILE)
        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')
        self.sess = sess
        (self.clip, self.clip_preprocess) = clip.load('ViT-B/32', device='cuda:0')
        self.prompt_engineering = True
        self.this_is = True
        self.temperature = 100.0
        self.use_softmax = False
        self.out_name = ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0']

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        return self.postprocess(self.forward(*args, **kwargs))

    def forward(self, img: np.array, category_names: str, **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Run the forward pass for a model.\n\n        Returns:\n            Dict[str, Any]: output from the model forward pass\n        '
        (roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info) = self.sess.run(self.out_name, feed_dict={'Placeholder:0': img})
        return_dict = {'roi_boxes': roi_boxes, 'roi_scores': roi_scores, 'detection_boxes': detection_boxes, 'scores_unused': scores_unused, 'box_outputs': box_outputs, 'detection_masks': detection_masks, 'visual_features': visual_features, 'image_info': image_info, 'category_names': category_names}
        return return_dict

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        ' Model specific postprocess and convert model output to\n        standard model outputs.\n\n        Args:\n            inputs:  input data\n\n        Return:\n            dict of results:  a dict containing outputs of model, each\n                output should have the standard output name.\n        '
        max_boxes_to_return = 25
        nms_threshold = 0.6
        min_rpn_score_thresh = 0.9
        min_box_area = 220
        roi_boxes = inputs['roi_boxes']
        roi_scores = inputs['roi_scores']
        detection_boxes = inputs['detection_boxes']
        scores_unused = inputs['scores_unused']
        box_outputs = inputs['box_outputs']
        detection_masks = inputs['detection_masks']
        visual_features = inputs['visual_features']
        image_info = inputs['image_info']
        category_names = inputs['category_names']
        category_names = [x.strip() for x in category_names.split(';')]
        category_names = ['background'] + category_names
        categories = [{'name': item, 'id': idx + 1} for (idx, item) in enumerate(category_names)]
        roi_boxes = np.squeeze(roi_boxes, axis=0)
        roi_scores = np.squeeze(roi_scores, axis=0)
        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)
        image_info = np.squeeze(image_info, axis=0)
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        rescaled_detection_boxes = detection_boxes / image_scale
        nmsed_indices = nms(detection_boxes, roi_scores, thresh=nms_threshold)
        box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])
        valid_indices = np.where(np.logical_and(np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices), np.logical_and(np.logical_not(np.all(roi_boxes == 0.0, axis=-1)), np.logical_and(roi_scores >= min_rpn_score_thresh, box_sizes > min_box_area))))[0]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_return, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_return, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_return, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_return, ...]
        text_features = self._build_text_embedings(categories)
        raw_scores = detection_visual_feat.dot(text_features.T)
        if self.use_softmax:
            scores_all = softmax(self.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores
        indices = np.argsort(-np.max(scores_all, axis=1))
        (ymin, xmin, ymax, xmax) = np.split(rescaled_detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate([xmin, ymin, xmax, ymax], axis=-1)
        n_boxes = processed_boxes.shape[0]
        categories = []
        bboxes = []
        scores = []
        labels = []
        for anno_idx in indices[0:int(n_boxes)]:
            anno_bbox = processed_boxes[anno_idx]
            anno_scores = scores_all[anno_idx]
            if np.argmax(anno_scores) == 0:
                continue
            bboxes.append(anno_bbox)
            scores.append(anno_scores[1:])
            categories.append(category_names[1:])
            labels.append(np.argmax(anno_scores) - 1)
        bboxes = np.vstack(bboxes)
        scores = np.vstack(scores)
        return (scores, categories, bboxes)

    def _build_text_embedings(self, categories):
        if False:
            while True:
                i = 10

        def processed_name(name, rm_dot=False):
            if False:
                print('Hello World!')
            res = name.replace('_', ' ').replace('/', ' or ').lower()
            if rm_dot:
                res = res.rstrip('.')
            return res

        def article(name):
            if False:
                for i in range(10):
                    print('nop')
            return 'an' if name[0] in 'aeiou' else 'a'
        templates = multiple_templates
        run_on_gpu = torch.cuda.is_available()
        with torch.no_grad():
            all_text_embeddings = []
            for category in categories:
                texts = [template.format(processed_name(category['name'], rm_dot=True), article=article(category['name'])) for template in templates]
                if self.this_is:
                    texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text for text in texts]
                texts = clip.tokenize(texts)
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = self.clip.encode_text(texts)
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            if run_on_gpu:
                all_text_embeddings = all_text_embeddings.cuda()
        return all_text_embeddings.cpu().numpy().T
multiple_templates = ['There is {article} {} in the scene.', 'There is the {} in the scene.', 'a photo of {article} {} in the scene.', 'a photo of the {} in the scene.', 'a photo of one {} in the scene.', 'itap of {article} {}.', 'itap of my {}.', 'itap of the {}.', 'a photo of {article} {}.', 'a photo of my {}.', 'a photo of the {}.', 'a photo of one {}.', 'a photo of many {}.', 'a good photo of {article} {}.', 'a good photo of the {}.', 'a bad photo of {article} {}.', 'a bad photo of the {}.', 'a photo of a nice {}.', 'a photo of the nice {}.', 'a photo of a cool {}.', 'a photo of the cool {}.', 'a photo of a weird {}.', 'a photo of the weird {}.', 'a photo of a small {}.', 'a photo of the small {}.', 'a photo of a large {}.', 'a photo of the large {}.', 'a photo of a clean {}.', 'a photo of the clean {}.', 'a photo of a dirty {}.', 'a photo of the dirty {}.', 'a bright photo of {article} {}.', 'a bright photo of the {}.', 'a dark photo of {article} {}.', 'a dark photo of the {}.', 'a photo of a hard to see {}.', 'a photo of the hard to see {}.', 'a low resolution photo of {article} {}.', 'a low resolution photo of the {}.', 'a cropped photo of {article} {}.', 'a cropped photo of the {}.', 'a close-up photo of {article} {}.', 'a close-up photo of the {}.', 'a jpeg corrupted photo of {article} {}.', 'a jpeg corrupted photo of the {}.', 'a blurry photo of {article} {}.', 'a blurry photo of the {}.', 'a pixelated photo of {article} {}.', 'a pixelated photo of the {}.', 'a black and white photo of the {}.', 'a black and white photo of {article} {}.', 'a plastic {}.', 'the plastic {}.', 'a toy {}.', 'the toy {}.', 'a plushie {}.', 'the plushie {}.', 'a cartoon {}.', 'the cartoon {}.', 'an embroidered {}.', 'the embroidered {}.', 'a painting of the {}.', 'a painting of a {}.']

def nms(dets, scores, thresh, max_dets=1000):
    if False:
        return 10
    'Non-maximum suppression.\n    Args:\n        dets: [N, 4]\n        scores: [N,]\n        thresh: iou threshold. Float\n        max_dets: int.\n    '
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)
        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep