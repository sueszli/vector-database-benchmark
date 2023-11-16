"""Data parser and processing.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""
import tensorflow.compat.v2 as tf
from official.vision.detection.dataloader import anchor
from official.vision.detection.dataloader import mode_keys as ModeKeys
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.utils import autoaugment_utils
from official.vision.detection.utils import box_utils
from official.vision.detection.utils import input_utils

def process_source_id(source_id):
    if False:
        print('Hello World!')
    'Processes source_id to the right format.'
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int32)
    with tf.control_dependencies([source_id]):
        source_id = tf.cond(pred=tf.equal(tf.size(input=source_id), 0), true_fn=lambda : tf.cast(tf.constant(-1), tf.int32), false_fn=lambda : tf.identity(source_id))
    return source_id

def pad_groundtruths_to_fixed_size(gt, n):
    if False:
        return 10
    'Pads the first dimension of groundtruths labels to the fixed size.'
    gt['boxes'] = input_utils.pad_to_fixed_size(gt['boxes'], n, -1)
    gt['is_crowds'] = input_utils.pad_to_fixed_size(gt['is_crowds'], n, 0)
    gt['areas'] = input_utils.pad_to_fixed_size(gt['areas'], n, -1)
    gt['classes'] = input_utils.pad_to_fixed_size(gt['classes'], n, -1)
    return gt

class Parser(object):
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, output_size, min_level, max_level, num_scales, aspect_ratios, anchor_size, match_threshold=0.5, unmatched_threshold=0.5, aug_rand_hflip=False, aug_scale_min=1.0, aug_scale_max=1.0, use_autoaugment=False, autoaugment_policy_name='v0', skip_crowd_during_training=True, max_num_instances=100, use_bfloat16=True, mode=None):
        if False:
            return 10
        'Initializes parameters for parsing annotations in the dataset.\n\n    Args:\n      output_size: `Tensor` or `list` for [height, width] of output image. The\n        output_size should be divided by the largest feature stride 2^max_level.\n      min_level: `int` number of minimum level of the output feature pyramid.\n      max_level: `int` number of maximum level of the output feature pyramid.\n      num_scales: `int` number representing intermediate scales added\n        on each level. For instances, num_scales=2 adds one additional\n        intermediate anchor scales [2^0, 2^0.5] on each level.\n      aspect_ratios: `list` of float numbers representing the aspect raito\n        anchors added on each level. The number indicates the ratio of width to\n        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors\n        on each scale level.\n      anchor_size: `float` number representing the scale of size of the base\n        anchor to the feature stride 2^level.\n      match_threshold: `float` number between 0 and 1 representing the\n        lower-bound threshold to assign positive labels for anchors. An anchor\n        with a score over the threshold is labeled positive.\n      unmatched_threshold: `float` number between 0 and 1 representing the\n        upper-bound threshold to assign negative labels for anchors. An anchor\n        with a score below the threshold is labeled negative.\n      aug_rand_hflip: `bool`, if True, augment training with random\n        horizontal flip.\n      aug_scale_min: `float`, the minimum scale applied to `output_size` for\n        data augmentation during training.\n      aug_scale_max: `float`, the maximum scale applied to `output_size` for\n        data augmentation during training.\n      use_autoaugment: `bool`, if True, use the AutoAugment augmentation policy\n        during training.\n      autoaugment_policy_name: `string` that specifies the name of the\n        AutoAugment policy that will be used during training.\n      skip_crowd_during_training: `bool`, if True, skip annotations labeled with\n        `is_crowd` equals to 1.\n      max_num_instances: `int` number of maximum number of instances in an\n        image. The groundtruth data will be padded to `max_num_instances`.\n      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.\n      mode: a ModeKeys. Specifies if this is training, evaluation, prediction\n        or prediction with groundtruths in the outputs.\n    '
        self._mode = mode
        self._max_num_instances = max_num_instances
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = mode == ModeKeys.TRAIN
        self._example_decoder = tf_example_decoder.TfExampleDecoder(include_mask=False)
        self._output_size = output_size
        self._min_level = min_level
        self._max_level = max_level
        self._num_scales = num_scales
        self._aspect_ratios = aspect_ratios
        self._anchor_size = anchor_size
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold
        self._aug_rand_hflip = aug_rand_hflip
        self._aug_scale_min = aug_scale_min
        self._aug_scale_max = aug_scale_max
        self._use_autoaugment = use_autoaugment
        self._autoaugment_policy_name = autoaugment_policy_name
        self._use_bfloat16 = use_bfloat16
        if mode == ModeKeys.TRAIN:
            self._parse_fn = self._parse_train_data
        elif mode == ModeKeys.EVAL:
            self._parse_fn = self._parse_eval_data
        elif mode == ModeKeys.PREDICT or mode == ModeKeys.PREDICT_WITH_GT:
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Parses data to an image and associated training labels.\n\n    Args:\n      value: a string tensor holding a serialized tf.Example proto.\n\n    Returns:\n      image: image tensor that is preproessed to have normalized value and\n        dimension [output_size[0], output_size[1], 3]\n      labels:\n        cls_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location]. The height_l and\n          width_l represent the dimension of class logits at l-th level.\n        box_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location * 4]. The height_l and\n          width_l represent the dimension of bounding box regression output at\n          l-th level.\n        num_positives: number of positive anchors in the image.\n        anchor_boxes: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, 4] representing anchor boxes at each level.\n        image_info: a 2D `Tensor` that encodes the information of the image and\n          the applied preprocessing. It is in the format of\n          [[original_height, original_width], [scaled_height, scaled_width],\n           [y_scale, x_scale], [y_offset, x_offset]].\n        groundtruths:\n          source_id: source image id. Default value -1 if the source id is empty\n            in the groundtruth annotation.\n          boxes: groundtruth bounding box annotations. The box is represented in\n            [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed\n            dimension [self._max_num_instances, 4].\n          classes: groundtruth classes annotations. The tennsor is padded with\n            -1 to the fixed dimension [self._max_num_instances].\n          areas: groundtruth areas annotations. The tennsor is padded with -1\n            to the fixed dimension [self._max_num_instances].\n          is_crowds: groundtruth annotations to indicate if an annotation\n            represents a group of instances by value {0, 1}. The tennsor is\n            padded with 0 to the fixed dimension [self._max_num_instances].\n    '
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse_train_data(self, data):
        if False:
            i = 10
            return i + 15
        'Parses data for training and evaluation.'
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        is_crowds = data['groundtruth_is_crowd']
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(input=classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(pred=tf.greater(tf.size(input=is_crowds), 0), true_fn=lambda : tf.where(tf.logical_not(is_crowds))[:, 0], false_fn=lambda : tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
        image = data['image']
        if self._use_autoaugment:
            (image, boxes) = autoaugment_utils.distort_image_with_autoaugment(image, boxes, self._autoaugment_policy_name)
        image_shape = tf.shape(input=image)[0:2]
        image = input_utils.normalize_image(image)
        if self._aug_rand_hflip:
            (image, boxes) = input_utils.random_horizontal_flip(image, boxes)
        boxes = box_utils.denormalize_boxes(boxes, image_shape)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, padded_size=input_utils.compute_padded_size(self._output_size, 2 ** self._max_level), aug_scale_min=self._aug_scale_min, aug_scale_max=self._aug_scale_max)
        (image_height, image_width, _) = image.get_shape().as_list()
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, (image_height, image_width), offset)
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices)
        classes = tf.gather(classes, indices)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, (image_height, image_width))
        anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)
        (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        labels = {'cls_targets': cls_targets, 'box_targets': box_targets, 'anchor_boxes': input_anchor.multilevel_boxes, 'num_positives': num_positives, 'image_info': image_info}
        return (image, labels)

    def _parse_eval_data(self, data):
        if False:
            return 10
        'Parses data for training and evaluation.'
        groundtruths = {}
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        image = data['image']
        image_shape = tf.shape(input=image)[0:2]
        image = input_utils.normalize_image(image)
        boxes = box_utils.denormalize_boxes(boxes, image_shape)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, padded_size=input_utils.compute_padded_size(self._output_size, 2 ** self._max_level), aug_scale_min=1.0, aug_scale_max=1.0)
        (image_height, image_width, _) = image.get_shape().as_list()
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, (image_height, image_width), offset)
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices)
        classes = tf.gather(classes, indices)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, (image_height, image_width))
        anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)
        (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        groundtruths = {'source_id': data['source_id'], 'num_groundtrtuhs': tf.shape(data['groundtruth_classes']), 'image_info': image_info, 'boxes': box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape), 'classes': data['groundtruth_classes'], 'areas': data['groundtruth_area'], 'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32)}
        groundtruths['source_id'] = process_source_id(groundtruths['source_id'])
        groundtruths = pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
        labels = {'cls_targets': cls_targets, 'box_targets': box_targets, 'anchor_boxes': input_anchor.multilevel_boxes, 'num_positives': num_positives, 'image_info': image_info, 'groundtruths': groundtruths}
        return (image, labels)

    def _parse_predict_data(self, data):
        if False:
            i = 10
            return i + 15
        'Parses data for prediction.'
        image = data['image']
        image_shape = tf.shape(input=image)[0:2]
        image = input_utils.normalize_image(image)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, padded_size=input_utils.compute_padded_size(self._output_size, 2 ** self._max_level), aug_scale_min=1.0, aug_scale_max=1.0)
        (image_height, image_width, _) = image.get_shape().as_list()
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, (image_height, image_width))
        labels = {'anchor_boxes': input_anchor.multilevel_boxes, 'image_info': image_info}
        if self._mode == ModeKeys.PREDICT_WITH_GT:
            boxes = box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape)
            groundtruths = {'source_id': data['source_id'], 'num_detections': tf.shape(data['groundtruth_classes']), 'boxes': boxes, 'classes': data['groundtruth_classes'], 'areas': data['groundtruth_area'], 'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32)}
            groundtruths['source_id'] = process_source_id(groundtruths['source_id'])
            groundtruths = pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
            labels['groundtruths'] = groundtruths
            classes = data['groundtruth_classes']
            image_scale = image_info[2, :]
            offset = image_info[3, :]
            boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, (image_height, image_width), offset)
            indices = box_utils.get_non_empty_box_indices(boxes)
            boxes = tf.gather(boxes, indices)
            anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)
            (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
            labels['cls_targets'] = cls_targets
            labels['box_targets'] = box_targets
            labels['num_positives'] = num_positives
        return (image, labels)