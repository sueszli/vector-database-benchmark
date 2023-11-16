"""Data parser and processing for Mask R-CNN."""
import tensorflow.compat.v2 as tf
from official.vision.detection.dataloader import anchor
from official.vision.detection.dataloader import mode_keys as ModeKeys
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.utils import box_utils
from official.vision.detection.utils import dataloader_utils
from official.vision.detection.utils import input_utils

class Parser(object):
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, output_size, min_level, max_level, num_scales, aspect_ratios, anchor_size, rpn_match_threshold=0.7, rpn_unmatched_threshold=0.3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5, aug_rand_hflip=False, aug_scale_min=1.0, aug_scale_max=1.0, skip_crowd_during_training=True, max_num_instances=100, include_mask=False, mask_crop_size=112, use_bfloat16=True, mode=None):
        if False:
            print('Hello World!')
        'Initializes parameters for parsing annotations in the dataset.\n\n    Args:\n      output_size: `Tensor` or `list` for [height, width] of output image. The\n        output_size should be divided by the largest feature stride 2^max_level.\n      min_level: `int` number of minimum level of the output feature pyramid.\n      max_level: `int` number of maximum level of the output feature pyramid.\n      num_scales: `int` number representing intermediate scales added\n        on each level. For instances, num_scales=2 adds one additional\n        intermediate anchor scales [2^0, 2^0.5] on each level.\n      aspect_ratios: `list` of float numbers representing the aspect raito\n        anchors added on each level. The number indicates the ratio of width to\n        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors\n        on each scale level.\n      anchor_size: `float` number representing the scale of size of the base\n        anchor to the feature stride 2^level.\n      rpn_match_threshold:\n      rpn_unmatched_threshold:\n      rpn_batch_size_per_im:\n      rpn_fg_fraction:\n      aug_rand_hflip: `bool`, if True, augment training with random\n        horizontal flip.\n      aug_scale_min: `float`, the minimum scale applied to `output_size` for\n        data augmentation during training.\n      aug_scale_max: `float`, the maximum scale applied to `output_size` for\n        data augmentation during training.\n      skip_crowd_during_training: `bool`, if True, skip annotations labeled with\n        `is_crowd` equals to 1.\n      max_num_instances: `int` number of maximum number of instances in an\n        image. The groundtruth data will be padded to `max_num_instances`.\n      include_mask: a bool to indicate whether parse mask groundtruth.\n      mask_crop_size: the size which groundtruth mask is cropped to.\n      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.\n      mode: a ModeKeys. Specifies if this is training, evaluation, prediction\n        or prediction with groundtruths in the outputs.\n    '
        self._mode = mode
        self._max_num_instances = max_num_instances
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = mode == ModeKeys.TRAIN
        self._example_decoder = tf_example_decoder.TfExampleDecoder(include_mask=include_mask)
        self._output_size = output_size
        self._min_level = min_level
        self._max_level = max_level
        self._num_scales = num_scales
        self._aspect_ratios = aspect_ratios
        self._anchor_size = anchor_size
        self._rpn_match_threshold = rpn_match_threshold
        self._rpn_unmatched_threshold = rpn_unmatched_threshold
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._rpn_fg_fraction = rpn_fg_fraction
        self._aug_rand_hflip = aug_rand_hflip
        self._aug_scale_min = aug_scale_min
        self._aug_scale_max = aug_scale_max
        self._include_mask = include_mask
        self._mask_crop_size = mask_crop_size
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
        "Parses data to an image and associated training labels.\n\n    Args:\n      value: a string tensor holding a serialized tf.Example proto.\n\n    Returns:\n      image, labels: if mode == ModeKeys.TRAIN. see _parse_train_data.\n      {'images': image, 'labels': labels}: if mode == ModeKeys.PREDICT\n        or ModeKeys.PREDICT_WITH_GT.\n    "
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse_train_data(self, data):
        if False:
            print('Hello World!')
        'Parses data for training.\n\n    Args:\n      data: the decoded tensor dictionary from TfExampleDecoder.\n\n    Returns:\n      image: image tensor that is preproessed to have normalized value and\n        dimension [output_size[0], output_size[1], 3]\n      labels: a dictionary of tensors used for training. The following describes\n        {key: value} pairs in the dictionary.\n        image_info: a 2D `Tensor` that encodes the information of the image and\n          the applied preprocessing. It is in the format of\n          [[original_height, original_width], [scaled_height, scaled_width],\n        anchor_boxes: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, 4] representing anchor boxes at each level.\n        rpn_score_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location]. The height_l and\n          width_l represent the dimension of class logits at l-th level.\n        rpn_box_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location * 4]. The height_l and\n          width_l represent the dimension of bounding box regression output at\n          l-th level.\n        gt_boxes: Groundtruth bounding box annotations. The box is represented\n           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled\n           image that is fed to the network. The tennsor is padded with -1 to\n           the fixed dimension [self._max_num_instances, 4].\n        gt_classes: Groundtruth classes annotations. The tennsor is padded\n          with -1 to the fixed dimension [self._max_num_instances].\n        gt_masks: groundtrugh masks cropped by the bounding box and\n          resized to a fixed size determined by mask_crop_size.\n    '
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        if self._include_mask:
            masks = data['groundtruth_instance_masks']
        is_crowds = data['groundtruth_is_crowd']
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(tf.greater(tf.size(is_crowds), 0), lambda : tf.where(tf.logical_not(is_crowds))[:, 0], lambda : tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
            if self._include_mask:
                masks = tf.gather(masks, indices)
        image = data['image']
        image_shape = tf.shape(image)[0:2]
        image = input_utils.normalize_image(image)
        if self._aug_rand_hflip:
            if self._include_mask:
                (image, boxes, masks) = input_utils.random_horizontal_flip(image, boxes, masks)
            else:
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
        if self._include_mask:
            masks = tf.gather(masks, indices)
            cropped_boxes = boxes + tf.cast(tf.tile(tf.expand_dims(offset, axis=0), [1, 2]), dtype=tf.float32)
            cropped_boxes = box_utils.normalize_boxes(cropped_boxes, image_info[1, :])
            num_masks = tf.shape(masks)[0]
            masks = tf.image.crop_and_resize(tf.expand_dims(masks, axis=-1), cropped_boxes, box_indices=tf.range(num_masks, dtype=tf.int32), crop_size=[self._mask_crop_size, self._mask_crop_size], method='bilinear')
            masks = tf.squeeze(masks, axis=-1)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, (image_height, image_width))
        anchor_labeler = anchor.RpnAnchorLabeler(input_anchor, self._rpn_match_threshold, self._rpn_unmatched_threshold, self._rpn_batch_size_per_im, self._rpn_fg_fraction)
        (rpn_score_targets, rpn_box_targets) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=-1), dtype=tf.float32))
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        labels = {'anchor_boxes': input_anchor.multilevel_boxes, 'image_info': image_info, 'rpn_score_targets': rpn_score_targets, 'rpn_box_targets': rpn_box_targets}
        labels['gt_boxes'] = input_utils.pad_to_fixed_size(boxes, self._max_num_instances, -1)
        labels['gt_classes'] = input_utils.pad_to_fixed_size(classes, self._max_num_instances, -1)
        if self._include_mask:
            labels['gt_masks'] = input_utils.pad_to_fixed_size(masks, self._max_num_instances, -1)
        return (image, labels)

    def _parse_eval_data(self, data):
        if False:
            print('Hello World!')
        'Parses data for evaluation.'
        raise NotImplementedError('Not implemented!')

    def _parse_predict_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        "Parses data for prediction.\n\n    Args:\n      data: the decoded tensor dictionary from TfExampleDecoder.\n\n    Returns:\n      A dictionary of {'images': image, 'labels': labels} where\n        image: image tensor that is preproessed to have normalized value and\n          dimension [output_size[0], output_size[1], 3]\n        labels: a dictionary of tensors used for training. The following\n          describes {key: value} pairs in the dictionary.\n          source_ids: Source image id. Default value -1 if the source id is\n            empty in the groundtruth annotation.\n          image_info: a 2D `Tensor` that encodes the information of the image\n            and the applied preprocessing. It is in the format of\n            [[original_height, original_width], [scaled_height, scaled_width],\n          anchor_boxes: ordered dictionary with keys\n            [min_level, min_level+1, ..., max_level]. The values are tensor with\n            shape [height_l, width_l, 4] representing anchor boxes at each\n            level.\n    "
        image = data['image']
        image_shape = tf.shape(image)[0:2]
        image = input_utils.normalize_image(image)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, padded_size=input_utils.compute_padded_size(self._output_size, 2 ** self._max_level), aug_scale_min=1.0, aug_scale_max=1.0)
        (image_height, image_width, _) = image.get_shape().as_list()
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, (image_height, image_width))
        labels = {'source_id': dataloader_utils.process_source_id(data['source_id']), 'anchor_boxes': input_anchor.multilevel_boxes, 'image_info': image_info}
        if self._mode == ModeKeys.PREDICT_WITH_GT:
            boxes = box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape)
            groundtruths = {'source_id': data['source_id'], 'height': data['height'], 'width': data['width'], 'num_detections': tf.shape(data['groundtruth_classes']), 'boxes': boxes, 'classes': data['groundtruth_classes'], 'areas': data['groundtruth_area'], 'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32)}
            groundtruths['source_id'] = dataloader_utils.process_source_id(groundtruths['source_id'])
            groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
            labels['groundtruths'] = groundtruths
        return (image, labels)