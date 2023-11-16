"""Data parser and processing.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for ShapeMask.

Weicheng Kuo, Anelia Angelova, Jitendra Malik, Tsung-Yi Lin
ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors.
arXiv:1904.03239.
"""
import tensorflow.compat.v2 as tf
from official.vision.detection.dataloader import anchor
from official.vision.detection.dataloader import mode_keys as ModeKeys
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.utils import box_utils
from official.vision.detection.utils import class_utils
from official.vision.detection.utils import dataloader_utils
from official.vision.detection.utils import input_utils

class Parser(object):
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, output_size, min_level, max_level, num_scales, aspect_ratios, anchor_size, use_category=True, outer_box_scale=1.0, box_jitter_scale=0.025, num_sampled_masks=8, mask_crop_size=32, mask_min_level=3, mask_max_level=5, upsample_factor=4, match_threshold=0.5, unmatched_threshold=0.5, aug_rand_hflip=False, aug_scale_min=1.0, aug_scale_max=1.0, skip_crowd_during_training=True, max_num_instances=100, use_bfloat16=True, mask_train_class='all', mode=None):
        if False:
            print('Hello World!')
        'Initializes parameters for parsing annotations in the dataset.\n\n    Args:\n      output_size: `Tensor` or `list` for [height, width] of output image. The\n        output_size should be divided by the largest feature stride 2^max_level.\n      min_level: `int` number of minimum level of the output feature pyramid.\n      max_level: `int` number of maximum level of the output feature pyramid.\n      num_scales: `int` number representing intermediate scales added\n        on each level. For instances, num_scales=2 adds one additional\n        intermediate anchor scales [2^0, 2^0.5] on each level.\n      aspect_ratios: `list` of float numbers representing the aspect raito\n        anchors added on each level. The number indicates the ratio of width to\n        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors\n        on each scale level.\n      anchor_size: `float` number representing the scale of size of the base\n        anchor to the feature stride 2^level.\n      use_category: if `False`, treat all object in all classes in one\n        foreground category.\n      outer_box_scale: `float` number in a range of [1.0, inf) representing\n        the scale from object box to outer box. The mask branch predicts\n        instance mask enclosed in outer box.\n      box_jitter_scale: `float` number representing the noise magnitude to\n        jitter the training groundtruth boxes for mask branch.\n      num_sampled_masks: `int` number of sampled masks for training.\n      mask_crop_size: `list` for [height, width] of output training masks.\n      mask_min_level: `int` number indicating the minimum feature level to\n        obtain instance features.\n      mask_max_level: `int` number indicating the maximum feature level to\n        obtain instance features.\n      upsample_factor: `int` factor of upsampling the fine mask predictions.\n      match_threshold: `float` number between 0 and 1 representing the\n        lower-bound threshold to assign positive labels for anchors. An anchor\n        with a score over the threshold is labeled positive.\n      unmatched_threshold: `float` number between 0 and 1 representing the\n        upper-bound threshold to assign negative labels for anchors. An anchor\n        with a score below the threshold is labeled negative.\n      aug_rand_hflip: `bool`, if True, augment training with random\n        horizontal flip.\n      aug_scale_min: `float`, the minimum scale applied to `output_size` for\n        data augmentation during training.\n      aug_scale_max: `float`, the maximum scale applied to `output_size` for\n        data augmentation during training.\n      skip_crowd_during_training: `bool`, if True, skip annotations labeled with\n        `is_crowd` equals to 1.\n      max_num_instances: `int` number of maximum number of instances in an\n        image. The groundtruth data will be padded to `max_num_instances`.\n      use_bfloat16: `bool`, if True, cast output image to tf.bfloat16.\n      mask_train_class: a string of experiment mode: `all`, `voc` or `nonvoc`.\n      mode: a ModeKeys. Specifies if this is training, evaluation, prediction\n        or prediction with groundtruths in the outputs.\n    '
        self._mode = mode
        self._mask_train_class = mask_train_class
        self._max_num_instances = max_num_instances
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = mode == ModeKeys.TRAIN
        self._example_decoder = tf_example_decoder.TfExampleDecoder(include_mask=True)
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
        self._use_bfloat16 = use_bfloat16
        self._use_category = use_category
        self._num_sampled_masks = num_sampled_masks
        self._mask_crop_size = mask_crop_size
        self._mask_min_level = mask_min_level
        self._mask_max_level = mask_max_level
        self._outer_box_scale = outer_box_scale
        self._box_jitter_scale = box_jitter_scale
        self._up_sample_factor = upsample_factor
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
            while True:
                i = 10
        'Parses data to an image and associated training labels.\n\n    Args:\n      value: a string tensor holding a serialized tf.Example proto.\n\n    Returns:\n      image: image tensor that is preproessed to have normalized value and\n        dimension [output_size[0], output_size[1], 3]\n      labels:\n        cls_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location]. The height_l and\n          width_l represent the dimension of class logits at l-th level.\n        box_targets: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, anchors_per_location * 4]. The height_l and\n          width_l represent the dimension of bounding box regression output at\n          l-th level.\n        num_positives: number of positive anchors in the image.\n        anchor_boxes: ordered dictionary with keys\n          [min_level, min_level+1, ..., max_level]. The values are tensor with\n          shape [height_l, width_l, 4] representing anchor boxes at each level.\n        image_scale: 2D float `Tensor` representing scale factors that apply\n          to [height, width] of input image.\n        mask_boxes: sampled boxes that tightly enclose the training masks. The\n          box is represented in [y1, x1, y2, x2] format. The tensor is sampled\n          to the fixed dimension [self._num_sampled_masks, 4].\n        mask_outer_boxes: loose box that enclose sampled tight box. The\n          box is represented in [y1, x1, y2, x2] format. The tensor is sampled\n          to the fixed dimension [self._num_sampled_masks, 4].\n        mask_targets: training binary mask targets. The tensor has shape\n          [self._num_sampled_masks, self._mask_crop_size, self._mask_crop_size].\n        mask_classes: the class ids of sampled training masks. The tensor has\n          shape [self._num_sampled_masks].\n        mask_is_valid: the binary tensor to indicate if the sampled masks are\n          valide. The sampled masks are invalid when no mask annotations are\n          included in the image. The tensor has shape [1].\n        groundtruths:\n          source_id: source image id. Default value -1 if the source id is empty\n            in the groundtruth annotation.\n          boxes: groundtruth bounding box annotations. The box is represented in\n            [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed\n            dimension [self._max_num_instances, 4].\n          classes: groundtruth classes annotations. The tensor is padded with\n            -1 to the fixed dimension [self._max_num_instances].\n          areas: groundtruth areas annotations. The tensor is padded with -1\n            to the fixed dimension [self._max_num_instances].\n          is_crowds: groundtruth annotations to indicate if an annotation\n            represents a group of instances by value {0, 1}. The tensor is\n            padded with 0 to the fixed dimension [self._max_num_instances].\n    '
        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse_train_data(self, data):
        if False:
            i = 10
            return i + 15
        'Parse data for ShapeMask training.'
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        masks = data['groundtruth_instance_masks']
        is_crowds = data['groundtruth_is_crowd']
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(tf.greater(tf.size(is_crowds), 0), lambda : tf.where(tf.logical_not(is_crowds))[:, 0], lambda : tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices)
            boxes = tf.gather(boxes, indices)
            masks = tf.gather(masks, indices)
        image = data['image']
        image_shape = tf.shape(image)[0:2]
        if not self._use_category:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
        image = input_utils.normalize_image(image)
        if self._aug_rand_hflip:
            (image, boxes, masks) = input_utils.random_horizontal_flip(image, boxes, masks)
        boxes = box_utils.denormalize_boxes(boxes, image_shape)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, self._output_size, aug_scale_min=self._aug_scale_min, aug_scale_max=self._aug_scale_max)
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, self._output_size, offset)
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices)
        classes = tf.gather(classes, indices)
        masks = tf.gather(masks, indices)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, self._output_size)
        anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)
        (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
        num_masks = tf.shape(masks)[0]
        mask_shape = tf.shape(masks)[1:3]
        padded_boxes = input_utils.pad_to_fixed_size(boxes, self._num_sampled_masks)
        padded_classes = input_utils.pad_to_fixed_size(classes, self._num_sampled_masks)
        padded_masks = input_utils.pad_to_fixed_size(masks, self._num_sampled_masks)
        rand_indices = tf.random.shuffle(tf.range(tf.maximum(num_masks, self._num_sampled_masks)))
        rand_indices = tf.math.mod(rand_indices, tf.maximum(num_masks, 1))
        rand_indices = rand_indices[0:self._num_sampled_masks]
        rand_indices = tf.reshape(rand_indices, [self._num_sampled_masks])
        sampled_boxes = tf.gather(padded_boxes, rand_indices)
        sampled_classes = tf.gather(padded_classes, rand_indices)
        sampled_masks = tf.gather(padded_masks, rand_indices)
        sampled_boxes = box_utils.jitter_boxes(sampled_boxes, noise_scale=self._box_jitter_scale)
        sampled_boxes = box_utils.clip_boxes(sampled_boxes, self._output_size)
        mask_outer_boxes = box_utils.compute_outer_boxes(sampled_boxes, tf.shape(image)[0:2], scale=self._outer_box_scale)
        mask_outer_boxes = box_utils.clip_boxes(mask_outer_boxes, self._output_size)
        mask_outer_boxes_ori = mask_outer_boxes
        mask_outer_boxes_ori += tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
        mask_outer_boxes_ori /= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        norm_mask_outer_boxes_ori = box_utils.normalize_boxes(mask_outer_boxes_ori, mask_shape)
        sampled_masks = tf.cast(tf.expand_dims(sampled_masks, axis=-1), tf.float32)
        mask_targets = tf.image.crop_and_resize(sampled_masks, norm_mask_outer_boxes_ori, box_indices=tf.range(self._num_sampled_masks), crop_size=[self._mask_crop_size, self._mask_crop_size], method='bilinear', extrapolation_value=0, name='train_mask_targets')
        mask_targets = tf.where(tf.greater_equal(mask_targets, 0.5), tf.ones_like(mask_targets), tf.zeros_like(mask_targets))
        mask_targets = tf.squeeze(mask_targets, axis=-1)
        if self._up_sample_factor > 1:
            fine_mask_targets = tf.image.crop_and_resize(sampled_masks, norm_mask_outer_boxes_ori, box_indices=tf.range(self._num_sampled_masks), crop_size=[self._mask_crop_size * self._up_sample_factor, self._mask_crop_size * self._up_sample_factor], method='bilinear', extrapolation_value=0, name='train_mask_targets')
            fine_mask_targets = tf.where(tf.greater_equal(fine_mask_targets, 0.5), tf.ones_like(fine_mask_targets), tf.zeros_like(fine_mask_targets))
            fine_mask_targets = tf.squeeze(fine_mask_targets, axis=-1)
        else:
            fine_mask_targets = mask_targets
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        valid_image = tf.cast(tf.not_equal(num_masks, 0), tf.int32)
        if self._mask_train_class == 'all':
            mask_is_valid = valid_image * tf.ones_like(sampled_classes, tf.int32)
        else:
            mask_valid_classes = tf.cast(tf.expand_dims(class_utils.coco_split_class_ids(self._mask_train_class), 1), sampled_classes.dtype)
            match = tf.reduce_any(tf.equal(tf.expand_dims(sampled_classes, 0), mask_valid_classes), 0)
            mask_is_valid = valid_image * tf.cast(match, tf.int32)
        labels = {'cls_targets': cls_targets, 'box_targets': box_targets, 'anchor_boxes': input_anchor.multilevel_boxes, 'num_positives': num_positives, 'image_info': image_info, 'mask_boxes': sampled_boxes, 'mask_outer_boxes': mask_outer_boxes, 'mask_targets': mask_targets, 'fine_mask_targets': fine_mask_targets, 'mask_classes': sampled_classes, 'mask_is_valid': mask_is_valid}
        return (image, labels)

    def _parse_predict_data(self, data):
        if False:
            print('Hello World!')
        'Parse data for ShapeMask training.'
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        masks = data['groundtruth_instance_masks']
        image = data['image']
        image_shape = tf.shape(image)[0:2]
        if not self._use_category:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
        image = input_utils.normalize_image(image)
        boxes = box_utils.denormalize_boxes(boxes, image_shape)
        (image, image_info) = input_utils.resize_and_crop_image(image, self._output_size, self._output_size, aug_scale_min=1.0, aug_scale_max=1.0)
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, self._output_size, offset)
        masks = input_utils.resize_and_crop_masks(tf.expand_dims(masks, axis=-1), image_scale, self._output_size, offset)
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices)
        classes = tf.gather(classes, indices)
        input_anchor = anchor.Anchor(self._min_level, self._max_level, self._num_scales, self._aspect_ratios, self._anchor_size, self._output_size)
        anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)
        if self._use_bfloat16:
            image = tf.cast(image, dtype=tf.bfloat16)
        labels = {'anchor_boxes': input_anchor.multilevel_boxes, 'image_info': image_info}
        if self._mode == ModeKeys.PREDICT_WITH_GT:
            groundtruths = {'source_id': data['source_id'], 'num_detections': tf.shape(data['groundtruth_classes']), 'boxes': box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape), 'classes': data['groundtruth_classes'], 'areas': data['groundtruth_area'], 'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32)}
            groundtruths['source_id'] = dataloader_utils.process_source_id(groundtruths['source_id'])
            groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
            (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))
            labels.update({'cls_targets': cls_targets, 'box_targets': box_targets, 'num_positives': num_positives, 'groundtruths': groundtruths})
        return (image, labels)