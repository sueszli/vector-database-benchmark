"""SSD Meta-architecture definition.

General tensorflow implementation of convolutional Multibox/SSD detection
models.
"""
import abc
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib import tpu as contrib_tpu
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils
slim = contrib_slim

class SSDFeatureExtractor(object):
    """SSD Slim Feature Extractor definition."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, num_layers=6, override_base_feature_extractor_hyperparams=False):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d\n        and separable_conv2d ops in the layers that are added on top of the\n        base feature extractor.\n      reuse_weights: whether to reuse variables. Default is None.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      num_layers: Number of SSD layers.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n    '
        self._is_training = is_training
        self._depth_multiplier = depth_multiplier
        self._min_depth = min_depth
        self._pad_to_multiple = pad_to_multiple
        self._conv_hyperparams_fn = conv_hyperparams_fn
        self._reuse_weights = reuse_weights
        self._use_explicit_padding = use_explicit_padding
        self._use_depthwise = use_depthwise
        self._num_layers = num_layers
        self._override_base_feature_extractor_hyperparams = override_base_feature_extractor_hyperparams

    @property
    def is_keras_model(self):
        if False:
            return 10
        return False

    @abc.abstractmethod
    def preprocess(self, resized_inputs):
        if False:
            i = 10
            return i + 15
        'Preprocesses images for feature extraction (minus image resizing).\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros.\n    '
        pass

    @abc.abstractmethod
    def extract_features(self, preprocessed_inputs):
        if False:
            while True:
                i = 10
        'Extracts features from preprocessed inputs.\n\n    This function is responsible for extracting feature maps from preprocessed\n    images.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    '
        raise NotImplementedError

    def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
        if False:
            i = 10
            return i + 15
        'Returns a map of variables to load from a foreign checkpoint.\n\n    Args:\n      feature_extractor_scope: A scope name for the feature extractor.\n\n    Returns:\n      A dict mapping variable names (to load from a checkpoint) to variables in\n      the model graph.\n    '
        variables_to_restore = {}
        for variable in variables_helper.get_global_variables_safely():
            var_name = variable.op.name
            if var_name.startswith(feature_extractor_scope + '/'):
                var_name = var_name.replace(feature_extractor_scope + '/', '')
                variables_to_restore[var_name] = variable
        return variables_to_restore

class SSDKerasFeatureExtractor(tf.keras.Model):
    """SSD Feature Extractor definition."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams, freeze_batchnorm, inplace_batchnorm_update, use_explicit_padding=False, use_depthwise=False, num_layers=6, override_base_feature_extractor_hyperparams=False, name=None):
        if False:
            return 10
        "Constructor.\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams: `hyperparams_builder.KerasLayerHyperparams` object\n        containing convolution hyperparameters for the layers added on top of\n        the base feature extractor.\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      inplace_batchnorm_update: Whether to update batch norm moving average\n        values inplace. When this is false train op must add a control\n        dependency on tf.graphkeys.UPDATE_OPS collection in order to update\n        batch norm statistics.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      num_layers: Number of SSD layers.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_config`.\n      name: A string name scope to assign to the model. If 'None', Keras\n        will auto-generate one from the class name.\n    "
        super(SSDKerasFeatureExtractor, self).__init__(name=name)
        self._is_training = is_training
        self._depth_multiplier = depth_multiplier
        self._min_depth = min_depth
        self._pad_to_multiple = pad_to_multiple
        self._conv_hyperparams = conv_hyperparams
        self._freeze_batchnorm = freeze_batchnorm
        self._inplace_batchnorm_update = inplace_batchnorm_update
        self._use_explicit_padding = use_explicit_padding
        self._use_depthwise = use_depthwise
        self._num_layers = num_layers
        self._override_base_feature_extractor_hyperparams = override_base_feature_extractor_hyperparams

    @property
    def is_keras_model(self):
        if False:
            return 10
        return True

    @abc.abstractmethod
    def preprocess(self, resized_inputs):
        if False:
            for i in range(10):
                print('nop')
        'Preprocesses images for feature extraction (minus image resizing).\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def _extract_features(self, preprocessed_inputs):
        if False:
            for i in range(10):
                print('nop')
        'Extracts features from preprocessed inputs.\n\n    This function is responsible for extracting feature maps from preprocessed\n    images.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    '
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._extract_features(inputs)

    def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
        if False:
            print('Hello World!')
        'Returns a map of variables to load from a foreign checkpoint.\n\n    Args:\n      feature_extractor_scope: A scope name for the feature extractor.\n\n    Returns:\n      A dict mapping variable names (to load from a checkpoint) to variables in\n      the model graph.\n    '
        variables_to_restore = {}
        if tf.executing_eagerly():
            for variable in self.variables:
                var_name = variable.name[:-2]
                if var_name.startswith(feature_extractor_scope + '/'):
                    var_name = var_name.replace(feature_extractor_scope + '/', '')
                variables_to_restore[var_name] = variable
        else:
            for variable in variables_helper.get_global_variables_safely():
                var_name = variable.op.name
                if var_name.startswith(feature_extractor_scope + '/'):
                    var_name = var_name.replace(feature_extractor_scope + '/', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore

class SSDMetaArch(model.DetectionModel):
    """SSD Meta-architecture definition."""

    def __init__(self, is_training, anchor_generator, box_predictor, box_coder, feature_extractor, encode_background_as_zeros, image_resizer_fn, non_max_suppression_fn, score_conversion_fn, classification_loss, localization_loss, classification_loss_weight, localization_loss_weight, normalize_loss_by_num_matches, hard_example_miner, target_assigner_instance, add_summaries=True, normalize_loc_loss_by_codesize=False, freeze_batchnorm=False, inplace_batchnorm_update=False, add_background_class=True, explicit_background_class=False, random_example_sampler=None, expected_loss_weights_fn=None, use_confidences_as_targets=False, implicit_example_weight=0.5, equalization_loss_config=None, return_raw_detections_during_predict=False, nms_on_host=True):
        if False:
            i = 10
            return i + 15
        'SSDMetaArch Constructor.\n\n    TODO(rathodv,jonathanhuang): group NMS parameters + score converter into\n    a class and loss parameters into a class and write config protos for\n    postprocessing and losses.\n\n    Args:\n      is_training: A boolean indicating whether the training version of the\n        computation graph should be constructed.\n      anchor_generator: an anchor_generator.AnchorGenerator object.\n      box_predictor: a box_predictor.BoxPredictor object.\n      box_coder: a box_coder.BoxCoder object.\n      feature_extractor: a SSDFeatureExtractor object.\n      encode_background_as_zeros: boolean determining whether background\n        targets are to be encoded as an all zeros vector or a one-hot\n        vector (where background is the 0th class).\n      image_resizer_fn: a callable for image resizing.  This callable always\n        takes a rank-3 image tensor (corresponding to a single image) and\n        returns a rank-3 image tensor, possibly with new spatial dimensions and\n        a 1-D tensor of shape [3] indicating shape of true image within\n        the resized image tensor as the resized image tensor could be padded.\n        See builders/image_resizer_builder.py.\n      non_max_suppression_fn: batch_multiclass_non_max_suppression\n        callable that takes `boxes`, `scores` and optional `clip_window`\n        inputs (with all other inputs already set) and returns a dictionary\n        hold tensors with keys: `detection_boxes`, `detection_scores`,\n        `detection_classes` and `num_detections`. See `post_processing.\n        batch_multiclass_non_max_suppression` for the type and shape of these\n        tensors.\n      score_conversion_fn: callable elementwise nonlinearity (that takes tensors\n        as inputs and returns tensors).  This is usually used to convert logits\n        to probabilities.\n      classification_loss: an object_detection.core.losses.Loss object.\n      localization_loss: a object_detection.core.losses.Loss object.\n      classification_loss_weight: float\n      localization_loss_weight: float\n      normalize_loss_by_num_matches: boolean\n      hard_example_miner: a losses.HardExampleMiner object (can be None)\n      target_assigner_instance: target_assigner.TargetAssigner instance to use.\n      add_summaries: boolean (default: True) controlling whether summary ops\n        should be added to tensorflow graph.\n      normalize_loc_loss_by_codesize: whether to normalize localization loss\n        by code size of the box encoder.\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      inplace_batchnorm_update: Whether to update batch norm moving average\n        values inplace. When this is false train op must add a control\n        dependency on tf.graphkeys.UPDATE_OPS collection in order to update\n        batch norm statistics.\n      add_background_class: Whether to add an implicit background class to\n        one-hot encodings of groundtruth labels. Set to false if training a\n        single class model or using groundtruth labels with an explicit\n        background class.\n      explicit_background_class: Set to true if using groundtruth labels with an\n        explicit background class, as in multiclass scores.\n      random_example_sampler: a BalancedPositiveNegativeSampler object that can\n        perform random example sampling when computing loss. If None, random\n        sampling process is skipped. Note that random example sampler and hard\n        example miner can both be applied to the model. In that case, random\n        sampler will take effect first and hard example miner can only process\n        the random sampled examples.\n      expected_loss_weights_fn: If not None, use to calculate\n        loss by background/foreground weighting. Should take batch_cls_targets\n        as inputs and return foreground_weights, background_weights. See\n        expected_classification_loss_by_expected_sampling and\n        expected_classification_loss_by_reweighting_unmatched_anchors in\n        third_party/tensorflow_models/object_detection/utils/ops.py as examples.\n      use_confidences_as_targets: Whether to use groundtruth_condifences field\n        to assign the targets.\n      implicit_example_weight: a float number that specifies the weight used\n        for the implicit negative examples.\n      equalization_loss_config: a namedtuple that specifies configs for\n        computing equalization loss.\n      return_raw_detections_during_predict: Whether to return raw detection\n        boxes in the predict() method. These are decoded boxes that have not\n        been through postprocessing (i.e. NMS). Default False.\n      nms_on_host: boolean (default: True) controlling whether NMS should be\n        carried out on the host (outside of TPU).\n    '
        super(SSDMetaArch, self).__init__(num_classes=box_predictor.num_classes)
        self._is_training = is_training
        self._freeze_batchnorm = freeze_batchnorm
        self._inplace_batchnorm_update = inplace_batchnorm_update
        self._anchor_generator = anchor_generator
        self._box_predictor = box_predictor
        self._box_coder = box_coder
        self._feature_extractor = feature_extractor
        self._add_background_class = add_background_class
        self._explicit_background_class = explicit_background_class
        if add_background_class and explicit_background_class:
            raise ValueError("Cannot have both 'add_background_class' and 'explicit_background_class' true.")
        if self._feature_extractor.is_keras_model:
            self._extract_features_scope = feature_extractor.name
        else:
            self._extract_features_scope = 'FeatureExtractor'
        if encode_background_as_zeros:
            background_class = [0]
        else:
            background_class = [1]
        if self._add_background_class:
            num_foreground_classes = self.num_classes
        else:
            num_foreground_classes = self.num_classes - 1
        self._unmatched_class_label = tf.constant(background_class + num_foreground_classes * [0], tf.float32)
        self._target_assigner = target_assigner_instance
        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
        self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize
        self._hard_example_miner = hard_example_miner
        self._random_example_sampler = random_example_sampler
        self._parallel_iterations = 16
        self._image_resizer_fn = image_resizer_fn
        self._non_max_suppression_fn = non_max_suppression_fn
        self._score_conversion_fn = score_conversion_fn
        self._anchors = None
        self._add_summaries = add_summaries
        self._batched_prediction_tensor_names = []
        self._expected_loss_weights_fn = expected_loss_weights_fn
        self._use_confidences_as_targets = use_confidences_as_targets
        self._implicit_example_weight = implicit_example_weight
        self._equalization_loss_config = equalization_loss_config
        self._return_raw_detections_during_predict = return_raw_detections_during_predict
        self._nms_on_host = nms_on_host

    @property
    def anchors(self):
        if False:
            return 10
        if not self._anchors:
            raise RuntimeError('anchors have not been constructed yet!')
        if not isinstance(self._anchors, box_list.BoxList):
            raise RuntimeError('anchors should be a BoxList object, but is not.')
        return self._anchors

    @property
    def batched_prediction_tensor_names(self):
        if False:
            print('Hello World!')
        if not self._batched_prediction_tensor_names:
            raise RuntimeError('Must call predict() method to get batched prediction tensor names.')
        return self._batched_prediction_tensor_names

    def preprocess(self, inputs):
        if False:
            i = 10
            return i + 15
        'Feature-extractor specific preprocessing.\n\n    SSD meta architecture uses a default clip_window of [0, 0, 1, 1] during\n    post-processing. On calling `preprocess` method, clip_window gets updated\n    based on `true_image_shapes` returned by `image_resizer_fn`.\n\n    Args:\n      inputs: a [batch, height_in, width_in, channels] float tensor representing\n        a batch of images with values between 0 and 255.0.\n\n    Returns:\n      preprocessed_inputs: a [batch, height_out, width_out, channels] float\n        tensor representing a batch of images.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros.\n\n    Raises:\n      ValueError: if inputs tensor does not have type tf.float32\n    '
        with tf.name_scope('Preprocessor'):
            (resized_inputs, true_image_shapes) = shape_utils.resize_images_and_return_shapes(inputs, self._image_resizer_fn)
            return (self._feature_extractor.preprocess(resized_inputs), true_image_shapes)

    def _compute_clip_window(self, preprocessed_images, true_image_shapes):
        if False:
            for i in range(10):
                print('nop')
        'Computes clip window to use during post_processing.\n\n    Computes a new clip window to use during post-processing based on\n    `resized_image_shapes` and `true_image_shapes` only if `preprocess` method\n    has been called. Otherwise returns a default clip window of [0, 0, 1, 1].\n\n    Args:\n      preprocessed_images: the [batch, height, width, channels] image\n          tensor.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros. Or None if the clip window should cover the full image.\n\n    Returns:\n      a 2-D float32 tensor of the form [batch_size, 4] containing the clip\n      window for each image in the batch in normalized coordinates (relative to\n      the resized dimensions) where each clip window is of the form [ymin, xmin,\n      ymax, xmax] or a default clip window of [0, 0, 1, 1].\n\n    '
        if true_image_shapes is None:
            return tf.constant([0, 0, 1, 1], dtype=tf.float32)
        resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(preprocessed_images)
        (true_heights, true_widths, _) = tf.unstack(tf.cast(true_image_shapes, dtype=tf.float32), axis=1)
        padded_height = tf.cast(resized_inputs_shape[1], dtype=tf.float32)
        padded_width = tf.cast(resized_inputs_shape[2], dtype=tf.float32)
        return tf.stack([tf.zeros_like(true_heights), tf.zeros_like(true_widths), true_heights / padded_height, true_widths / padded_width], axis=1)

    def predict(self, preprocessed_inputs, true_image_shapes):
        if False:
            i = 10
            return i + 15
        'Predicts unpostprocessed tensors from input tensor.\n\n    This function takes an input batch of images and runs it through the forward\n    pass of the network to yield unpostprocessesed predictions.\n\n    A side effect of calling the predict method is that self._anchors is\n    populated with a box_list.BoxList of anchors.  These anchors must be\n    constructed before the postprocess or loss functions can be called.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] image tensor.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros.\n\n    Returns:\n      prediction_dict: a dictionary holding "raw" prediction tensors:\n        1) preprocessed_inputs: the [batch, height, width, channels] image\n          tensor.\n        2) box_encodings: 4-D float tensor of shape [batch_size, num_anchors,\n          box_code_dimension] containing predicted boxes.\n        3) class_predictions_with_background: 3-D float tensor of shape\n          [batch_size, num_anchors, num_classes+1] containing class predictions\n          (logits) for each of the anchors.  Note that this tensor *includes*\n          background class predictions (at class index 0).\n        4) feature_maps: a list of tensors where the ith tensor has shape\n          [batch, height_i, width_i, depth_i].\n        5) anchors: 2-D float tensor of shape [num_anchors, 4] containing\n          the generated anchors in normalized coordinates.\n        6) final_anchors: 3-D float tensor of shape [batch_size, num_anchors, 4]\n          containing the generated anchors in normalized coordinates.\n        If self._return_raw_detections_during_predict is True, the dictionary\n        will also contain:\n        7) raw_detection_boxes: a 4-D float32 tensor with shape\n          [batch_size, self.max_num_proposals, 4] in normalized coordinates.\n        8) raw_detection_feature_map_indices: a 3-D int32 tensor with shape\n          [batch_size, self.max_num_proposals].\n    '
        if self._inplace_batchnorm_update:
            batchnorm_updates_collections = None
        else:
            batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS
        if self._feature_extractor.is_keras_model:
            feature_maps = self._feature_extractor(preprocessed_inputs)
        else:
            with slim.arg_scope([slim.batch_norm], is_training=self._is_training and (not self._freeze_batchnorm), updates_collections=batchnorm_updates_collections):
                with tf.variable_scope(None, self._extract_features_scope, [preprocessed_inputs]):
                    feature_maps = self._feature_extractor.extract_features(preprocessed_inputs)
        feature_map_spatial_dims = self._get_feature_map_spatial_dims(feature_maps)
        image_shape = shape_utils.combined_static_and_dynamic_shape(preprocessed_inputs)
        boxlist_list = self._anchor_generator.generate(feature_map_spatial_dims, im_height=image_shape[1], im_width=image_shape[2])
        self._anchors = box_list_ops.concatenate(boxlist_list)
        if self._box_predictor.is_keras_model:
            predictor_results_dict = self._box_predictor(feature_maps)
        else:
            with slim.arg_scope([slim.batch_norm], is_training=self._is_training and (not self._freeze_batchnorm), updates_collections=batchnorm_updates_collections):
                predictor_results_dict = self._box_predictor.predict(feature_maps, self._anchor_generator.num_anchors_per_location())
        predictions_dict = {'preprocessed_inputs': preprocessed_inputs, 'feature_maps': feature_maps, 'anchors': self._anchors.get(), 'final_anchors': tf.tile(tf.expand_dims(self._anchors.get(), 0), [image_shape[0], 1, 1])}
        for (prediction_key, prediction_list) in iter(predictor_results_dict.items()):
            prediction = tf.concat(prediction_list, axis=1)
            if prediction_key == 'box_encodings' and prediction.shape.ndims == 4 and (prediction.shape[2] == 1):
                prediction = tf.squeeze(prediction, axis=2)
            predictions_dict[prediction_key] = prediction
        if self._return_raw_detections_during_predict:
            predictions_dict.update(self._raw_detections_and_feature_map_inds(predictions_dict['box_encodings'], boxlist_list))
        self._batched_prediction_tensor_names = [x for x in predictions_dict if x != 'anchors']
        return predictions_dict

    def _raw_detections_and_feature_map_inds(self, box_encodings, boxlist_list):
        if False:
            return 10
        anchors = self._anchors.get()
        (raw_detection_boxes, _) = self._batch_decode(box_encodings, anchors)
        (batch_size, _, _) = shape_utils.combined_static_and_dynamic_shape(raw_detection_boxes)
        feature_map_indices = self._anchor_generator.anchor_index_to_feature_map_index(boxlist_list)
        feature_map_indices_batched = tf.tile(tf.expand_dims(feature_map_indices, 0), multiples=[batch_size, 1])
        return {fields.PredictionFields.raw_detection_boxes: raw_detection_boxes, fields.PredictionFields.raw_detection_feature_map_indices: feature_map_indices_batched}

    def _get_feature_map_spatial_dims(self, feature_maps):
        if False:
            for i in range(10):
                print('nop')
        'Return list of spatial dimensions for each feature map in a list.\n\n    Args:\n      feature_maps: a list of tensors where the ith tensor has shape\n          [batch, height_i, width_i, depth_i].\n\n    Returns:\n      a list of pairs (height, width) for each feature map in feature_maps\n    '
        feature_map_shapes = [shape_utils.combined_static_and_dynamic_shape(feature_map) for feature_map in feature_maps]
        return [(shape[1], shape[2]) for shape in feature_map_shapes]

    def postprocess(self, prediction_dict, true_image_shapes):
        if False:
            print('Hello World!')
        "Converts prediction tensors to final detections.\n\n    This function converts raw predictions tensors to final detection results by\n    slicing off the background class, decoding box predictions and applying\n    non max suppression and clipping to the image window.\n\n    See base class for output format conventions.  Note also that by default,\n    scores are to be interpreted as logits, but if a score_conversion_fn is\n    used, then scores are remapped (and may thus have a different\n    interpretation).\n\n    Args:\n      prediction_dict: a dictionary holding prediction tensors with\n        1) preprocessed_inputs: a [batch, height, width, channels] image\n          tensor.\n        2) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,\n          box_code_dimension] containing predicted boxes.\n        3) class_predictions_with_background: 3-D float tensor of shape\n          [batch_size, num_anchors, num_classes+1] containing class predictions\n          (logits) for each of the anchors.  Note that this tensor *includes*\n          background class predictions.\n        4) mask_predictions: (optional) a 5-D float tensor of shape\n          [batch_size, num_anchors, q, mask_height, mask_width]. `q` can be\n          either number of classes or 1 depending on whether a separate mask is\n          predicted per class.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros. Or None, if the clip window should cover the full image.\n\n    Returns:\n      detections: a dictionary containing the following fields\n        detection_boxes: [batch, max_detections, 4] tensor with post-processed\n          detection boxes.\n        detection_scores: [batch, max_detections] tensor with scalar scores for\n          post-processed detection boxes.\n        detection_multiclass_scores: [batch, max_detections,\n          num_classes_with_background] tensor with class score distribution for\n          post-processed detection boxes including background class if any.\n        detection_classes: [batch, max_detections] tensor with classes for\n          post-processed detection classes.\n        detection_keypoints: [batch, max_detections, num_keypoints, 2] (if\n          encoded in the prediction_dict 'box_encodings')\n        detection_masks: [batch_size, max_detections, mask_height, mask_width]\n          (optional)\n        num_detections: [batch]\n        raw_detection_boxes: [batch, total_detections, 4] tensor with decoded\n          detection boxes before Non-Max Suppression.\n        raw_detection_score: [batch, total_detections,\n          num_classes_with_background] tensor of multi-class scores for raw\n          detection boxes.\n    Raises:\n      ValueError: if prediction_dict does not contain `box_encodings` or\n        `class_predictions_with_background` fields.\n    "
        if 'box_encodings' not in prediction_dict or 'class_predictions_with_background' not in prediction_dict:
            raise ValueError('prediction_dict does not contain expected entries.')
        if 'anchors' not in prediction_dict:
            prediction_dict['anchors'] = self.anchors.get()
        with tf.name_scope('Postprocessor'):
            preprocessed_images = prediction_dict['preprocessed_inputs']
            box_encodings = prediction_dict['box_encodings']
            box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
            class_predictions_with_background = prediction_dict['class_predictions_with_background']
            (detection_boxes, detection_keypoints) = self._batch_decode(box_encodings, prediction_dict['anchors'])
            detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
            detection_boxes = tf.expand_dims(detection_boxes, axis=2)
            detection_scores_with_background = self._score_conversion_fn(class_predictions_with_background)
            detection_scores = tf.identity(detection_scores_with_background, 'raw_box_scores')
            if self._add_background_class or self._explicit_background_class:
                detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])
            additional_fields = None
            batch_size = shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0]
            if 'feature_maps' in prediction_dict:
                feature_map_list = []
                for feature_map in prediction_dict['feature_maps']:
                    feature_map_list.append(tf.reshape(feature_map, [batch_size, -1]))
                box_features = tf.concat(feature_map_list, 1)
                box_features = tf.identity(box_features, 'raw_box_features')
            additional_fields = {'multiclass_scores': detection_scores_with_background}
            if self._anchors is not None:
                num_boxes = self._anchors.num_boxes_static() or self._anchors.num_boxes()
                anchor_indices = tf.range(num_boxes)
                batch_anchor_indices = tf.tile(tf.expand_dims(anchor_indices, 0), [batch_size, 1])
                additional_fields.update({'anchor_indices': tf.cast(batch_anchor_indices, tf.float32)})
            if detection_keypoints is not None:
                detection_keypoints = tf.identity(detection_keypoints, 'raw_keypoint_locations')
                additional_fields[fields.BoxListFields.keypoints] = detection_keypoints
            with tf.init_scope():
                if tf.executing_eagerly():

                    def _non_max_suppression_wrapper(kwargs):
                        if False:
                            for i in range(10):
                                print('nop')
                        return self._non_max_suppression_fn(**kwargs)
                else:

                    def _non_max_suppression_wrapper(kwargs):
                        if False:
                            print('Hello World!')
                        if self._nms_on_host:
                            return contrib_tpu.outside_compilation(lambda x: self._non_max_suppression_fn(**x), kwargs)
                        else:
                            return self._non_max_suppression_fn(**kwargs)
            (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, nmsed_additional_fields, num_detections) = _non_max_suppression_wrapper({'boxes': detection_boxes, 'scores': detection_scores, 'clip_window': self._compute_clip_window(preprocessed_images, true_image_shapes), 'additional_fields': additional_fields, 'masks': prediction_dict.get('mask_predictions')})
            detection_dict = {fields.DetectionResultFields.detection_boxes: nmsed_boxes, fields.DetectionResultFields.detection_scores: nmsed_scores, fields.DetectionResultFields.detection_classes: nmsed_classes, fields.DetectionResultFields.detection_multiclass_scores: nmsed_additional_fields.get('multiclass_scores') if nmsed_additional_fields else None, fields.DetectionResultFields.num_detections: tf.cast(num_detections, dtype=tf.float32), fields.DetectionResultFields.raw_detection_boxes: tf.squeeze(detection_boxes, axis=2), fields.DetectionResultFields.raw_detection_scores: detection_scores_with_background}
            if nmsed_additional_fields is not None and 'anchor_indices' in nmsed_additional_fields:
                detection_dict.update({fields.DetectionResultFields.detection_anchor_indices: tf.cast(nmsed_additional_fields['anchor_indices'], tf.int32)})
            if nmsed_additional_fields is not None and fields.BoxListFields.keypoints in nmsed_additional_fields:
                detection_dict[fields.DetectionResultFields.detection_keypoints] = nmsed_additional_fields[fields.BoxListFields.keypoints]
            if nmsed_masks is not None:
                detection_dict[fields.DetectionResultFields.detection_masks] = nmsed_masks
            return detection_dict

    def loss(self, prediction_dict, true_image_shapes, scope=None):
        if False:
            return 10
        'Compute scalar loss tensors with respect to provided groundtruth.\n\n    Calling this function requires that groundtruth tensors have been\n    provided via the provide_groundtruth function.\n\n    Args:\n      prediction_dict: a dictionary holding prediction tensors with\n        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,\n          box_code_dimension] containing predicted boxes.\n        2) class_predictions_with_background: 3-D float tensor of shape\n          [batch_size, num_anchors, num_classes+1] containing class predictions\n          (logits) for each of the anchors. Note that this tensor *includes*\n          background class predictions.\n      true_image_shapes: int32 tensor of shape [batch, 3] where each row is\n        of the form [height, width, channels] indicating the shapes\n        of true images in the resized images, as resized images can be padded\n        with zeros.\n      scope: Optional scope name.\n\n    Returns:\n      a dictionary mapping loss keys (`localization_loss` and\n        `classification_loss`) to scalar tensors representing corresponding loss\n        values.\n    '
        with tf.name_scope(scope, 'Loss', prediction_dict.values()):
            keypoints = None
            if self.groundtruth_has_field(fields.BoxListFields.keypoints):
                keypoints = self.groundtruth_lists(fields.BoxListFields.keypoints)
            weights = None
            if self.groundtruth_has_field(fields.BoxListFields.weights):
                weights = self.groundtruth_lists(fields.BoxListFields.weights)
            confidences = None
            if self.groundtruth_has_field(fields.BoxListFields.confidences):
                confidences = self.groundtruth_lists(fields.BoxListFields.confidences)
            (batch_cls_targets, batch_cls_weights, batch_reg_targets, batch_reg_weights, batch_match) = self._assign_targets(self.groundtruth_lists(fields.BoxListFields.boxes), self.groundtruth_lists(fields.BoxListFields.classes), keypoints, weights, confidences)
            match_list = [matcher.Match(match) for match in tf.unstack(batch_match)]
            if self._add_summaries:
                self._summarize_target_assignment(self.groundtruth_lists(fields.BoxListFields.boxes), match_list)
            if self._random_example_sampler:
                batch_cls_per_anchor_weights = tf.reduce_mean(batch_cls_weights, axis=-1)
                batch_sampled_indicator = tf.cast(shape_utils.static_or_dynamic_map_fn(self._minibatch_subsample_fn, [batch_cls_targets, batch_cls_per_anchor_weights], dtype=tf.bool, parallel_iterations=self._parallel_iterations, back_prop=True), dtype=tf.float32)
                batch_reg_weights = tf.multiply(batch_sampled_indicator, batch_reg_weights)
                batch_cls_weights = tf.multiply(tf.expand_dims(batch_sampled_indicator, -1), batch_cls_weights)
            losses_mask = None
            if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
                losses_mask = tf.stack(self.groundtruth_lists(fields.InputDataFields.is_annotated))
            location_losses = self._localization_loss(prediction_dict['box_encodings'], batch_reg_targets, ignore_nan_targets=True, weights=batch_reg_weights, losses_mask=losses_mask)
            cls_losses = self._classification_loss(prediction_dict['class_predictions_with_background'], batch_cls_targets, weights=batch_cls_weights, losses_mask=losses_mask)
            if self._expected_loss_weights_fn:
                (batch_size, num_anchors, num_classes) = batch_cls_targets.get_shape()
                unmatched_targets = tf.ones([batch_size, num_anchors, 1]) * self._unmatched_class_label
                unmatched_cls_losses = self._classification_loss(prediction_dict['class_predictions_with_background'], unmatched_targets, weights=batch_cls_weights, losses_mask=losses_mask)
                if cls_losses.get_shape().ndims == 3:
                    (batch_size, num_anchors, num_classes) = cls_losses.get_shape()
                    cls_losses = tf.reshape(cls_losses, [batch_size, -1])
                    unmatched_cls_losses = tf.reshape(unmatched_cls_losses, [batch_size, -1])
                    batch_cls_targets = tf.reshape(batch_cls_targets, [batch_size, num_anchors * num_classes, -1])
                    batch_cls_targets = tf.concat([1 - batch_cls_targets, batch_cls_targets], axis=-1)
                    location_losses = tf.tile(location_losses, [1, num_classes])
                (foreground_weights, background_weights) = self._expected_loss_weights_fn(batch_cls_targets)
                cls_losses = foreground_weights * cls_losses + background_weights * unmatched_cls_losses
                location_losses *= foreground_weights
                classification_loss = tf.reduce_sum(cls_losses)
                localization_loss = tf.reduce_sum(location_losses)
            elif self._hard_example_miner:
                cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
                (localization_loss, classification_loss) = self._apply_hard_mining(location_losses, cls_losses, prediction_dict, match_list)
                if self._add_summaries:
                    self._hard_example_miner.summarize()
            else:
                cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
                localization_loss = tf.reduce_sum(location_losses)
                classification_loss = tf.reduce_sum(cls_losses)
            normalizer = tf.constant(1.0, dtype=tf.float32)
            if self._normalize_loss_by_num_matches:
                normalizer = tf.maximum(tf.cast(tf.reduce_sum(batch_reg_weights), dtype=tf.float32), 1.0)
            localization_loss_normalizer = normalizer
            if self._normalize_loc_loss_by_codesize:
                localization_loss_normalizer *= self._box_coder.code_size
            localization_loss = tf.multiply(self._localization_loss_weight / localization_loss_normalizer, localization_loss, name='localization_loss')
            classification_loss = tf.multiply(self._classification_loss_weight / normalizer, classification_loss, name='classification_loss')
            loss_dict = {'Loss/localization_loss': localization_loss, 'Loss/classification_loss': classification_loss}
        return loss_dict

    def _minibatch_subsample_fn(self, inputs):
        if False:
            while True:
                i = 10
        'Randomly samples anchors for one image.\n\n    Args:\n      inputs: a list of 2 inputs. First one is a tensor of shape [num_anchors,\n        num_classes] indicating targets assigned to each anchor. Second one\n        is a tensor of shape [num_anchors] indicating the class weight of each\n        anchor.\n\n    Returns:\n      batch_sampled_indicator: bool tensor of shape [num_anchors] indicating\n        whether the anchor should be selected for loss computation.\n    '
        (cls_targets, cls_weights) = inputs
        if self._add_background_class:
            background_class = tf.zeros_like(tf.slice(cls_targets, [0, 0], [-1, 1]))
            regular_class = tf.slice(cls_targets, [0, 1], [-1, -1])
            cls_targets = tf.concat([background_class, regular_class], 1)
        positives_indicator = tf.reduce_sum(cls_targets, axis=1)
        return self._random_example_sampler.subsample(tf.cast(cls_weights, tf.bool), batch_size=None, labels=tf.cast(positives_indicator, tf.bool))

    def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
        if False:
            print('Hello World!')
        positive_indices = tf.where(tf.greater(class_ids, 0))
        positive_anchor_cls_loss = tf.squeeze(tf.gather(cls_losses, positive_indices), axis=1)
        visualization_utils.add_cdf_image_summary(positive_anchor_cls_loss, 'PositiveAnchorLossCDF')
        negative_indices = tf.where(tf.equal(class_ids, 0))
        negative_anchor_cls_loss = tf.squeeze(tf.gather(cls_losses, negative_indices), axis=1)
        visualization_utils.add_cdf_image_summary(negative_anchor_cls_loss, 'NegativeAnchorLossCDF')

    def _assign_targets(self, groundtruth_boxes_list, groundtruth_classes_list, groundtruth_keypoints_list=None, groundtruth_weights_list=None, groundtruth_confidences_list=None):
        if False:
            print('Hello World!')
        'Assign groundtruth targets.\n\n    Adds a background class to each one-hot encoding of groundtruth classes\n    and uses target assigner to obtain regression and classification targets.\n\n    Args:\n      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]\n        containing coordinates of the groundtruth boxes.\n          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]\n          format and assumed to be normalized and clipped\n          relative to the image window with y_min <= y_max and x_min <= x_max.\n      groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of\n        shape [num_boxes, num_classes] containing the class targets with the 0th\n        index assumed to map to the first non-background class.\n      groundtruth_keypoints_list: (optional) a list of 3-D tensors of shape\n        [num_boxes, num_keypoints, 2]\n      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape\n        [num_boxes] containing weights for groundtruth boxes.\n      groundtruth_confidences_list: A list of 2-D tf.float32 tensors of shape\n        [num_boxes, num_classes] containing class confidences for\n        groundtruth boxes.\n\n    Returns:\n      batch_cls_targets: a tensor with shape [batch_size, num_anchors,\n        num_classes],\n      batch_cls_weights: a tensor with shape [batch_size, num_anchors],\n      batch_reg_targets: a tensor with shape [batch_size, num_anchors,\n        box_code_dimension]\n      batch_reg_weights: a tensor with shape [batch_size, num_anchors],\n      match_list: a list of matcher.Match objects encoding the match between\n        anchors and groundtruth boxes for each image of the batch,\n        with rows of the Match objects corresponding to groundtruth boxes\n        and columns corresponding to anchors.\n    '
        groundtruth_boxlists = [box_list.BoxList(boxes) for boxes in groundtruth_boxes_list]
        train_using_confidences = self._is_training and self._use_confidences_as_targets
        if self._add_background_class:
            groundtruth_classes_with_background_list = [tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT') for one_hot_encoding in groundtruth_classes_list]
            if train_using_confidences:
                groundtruth_confidences_with_background_list = [tf.pad(groundtruth_confidences, [[0, 0], [1, 0]], mode='CONSTANT') for groundtruth_confidences in groundtruth_confidences_list]
        else:
            groundtruth_classes_with_background_list = groundtruth_classes_list
        if groundtruth_keypoints_list is not None:
            for (boxlist, keypoints) in zip(groundtruth_boxlists, groundtruth_keypoints_list):
                boxlist.add_field(fields.BoxListFields.keypoints, keypoints)
        if train_using_confidences:
            return target_assigner.batch_assign_confidences(self._target_assigner, self.anchors, groundtruth_boxlists, groundtruth_confidences_with_background_list, groundtruth_weights_list, self._unmatched_class_label, self._add_background_class, self._implicit_example_weight)
        else:
            return target_assigner.batch_assign_targets(self._target_assigner, self.anchors, groundtruth_boxlists, groundtruth_classes_with_background_list, self._unmatched_class_label, groundtruth_weights_list)

    def _summarize_target_assignment(self, groundtruth_boxes_list, match_list):
        if False:
            while True:
                i = 10
        'Creates tensorflow summaries for the input boxes and anchors.\n\n    This function creates four summaries corresponding to the average\n    number (over images in a batch) of (1) groundtruth boxes, (2) anchors\n    marked as positive, (3) anchors marked as negative, and (4) anchors marked\n    as ignored.\n\n    Args:\n      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]\n        containing corners of the groundtruth boxes.\n      match_list: a list of matcher.Match objects encoding the match between\n        anchors and groundtruth boxes for each image of the batch,\n        with rows of the Match objects corresponding to groundtruth boxes\n        and columns corresponding to anchors.\n    '
        try:
            with tf.compat.v2.init_scope():
                if tf.compat.v2.executing_eagerly():
                    return
        except AttributeError:
            pass
        avg_num_gt_boxes = tf.reduce_mean(tf.cast(tf.stack([tf.shape(x)[0] for x in groundtruth_boxes_list]), dtype=tf.float32))
        avg_num_matched_gt_boxes = tf.reduce_mean(tf.cast(tf.stack([match.num_matched_rows() for match in match_list]), dtype=tf.float32))
        avg_pos_anchors = tf.reduce_mean(tf.cast(tf.stack([match.num_matched_columns() for match in match_list]), dtype=tf.float32))
        avg_neg_anchors = tf.reduce_mean(tf.cast(tf.stack([match.num_unmatched_columns() for match in match_list]), dtype=tf.float32))
        avg_ignored_anchors = tf.reduce_mean(tf.cast(tf.stack([match.num_ignored_columns() for match in match_list]), dtype=tf.float32))
        tf.summary.scalar('AvgNumGroundtruthBoxesPerImage', avg_num_gt_boxes, family='TargetAssignment')
        tf.summary.scalar('AvgNumGroundtruthBoxesMatchedPerImage', avg_num_matched_gt_boxes, family='TargetAssignment')
        tf.summary.scalar('AvgNumPositiveAnchorsPerImage', avg_pos_anchors, family='TargetAssignment')
        tf.summary.scalar('AvgNumNegativeAnchorsPerImage', avg_neg_anchors, family='TargetAssignment')
        tf.summary.scalar('AvgNumIgnoredAnchorsPerImage', avg_ignored_anchors, family='TargetAssignment')

    def _apply_hard_mining(self, location_losses, cls_losses, prediction_dict, match_list):
        if False:
            for i in range(10):
                print('nop')
        'Applies hard mining to anchorwise losses.\n\n    Args:\n      location_losses: Float tensor of shape [batch_size, num_anchors]\n        representing anchorwise location losses.\n      cls_losses: Float tensor of shape [batch_size, num_anchors]\n        representing anchorwise classification losses.\n      prediction_dict: p a dictionary holding prediction tensors with\n        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,\n          box_code_dimension] containing predicted boxes.\n        2) class_predictions_with_background: 3-D float tensor of shape\n          [batch_size, num_anchors, num_classes+1] containing class predictions\n          (logits) for each of the anchors.  Note that this tensor *includes*\n          background class predictions.\n        3) anchors: (optional) 2-D float tensor of shape [num_anchors, 4].\n      match_list: a list of matcher.Match objects encoding the match between\n        anchors and groundtruth boxes for each image of the batch,\n        with rows of the Match objects corresponding to groundtruth boxes\n        and columns corresponding to anchors.\n\n    Returns:\n      mined_location_loss: a float scalar with sum of localization losses from\n        selected hard examples.\n      mined_cls_loss: a float scalar with sum of classification losses from\n        selected hard examples.\n    '
        class_predictions = prediction_dict['class_predictions_with_background']
        if self._add_background_class:
            class_predictions = tf.slice(class_predictions, [0, 0, 1], [-1, -1, -1])
        if 'anchors' not in prediction_dict:
            prediction_dict['anchors'] = self.anchors.get()
        (decoded_boxes, _) = self._batch_decode(prediction_dict['box_encodings'], prediction_dict['anchors'])
        decoded_box_tensors_list = tf.unstack(decoded_boxes)
        class_prediction_list = tf.unstack(class_predictions)
        decoded_boxlist_list = []
        for (box_location, box_score) in zip(decoded_box_tensors_list, class_prediction_list):
            decoded_boxlist = box_list.BoxList(box_location)
            decoded_boxlist.add_field('scores', box_score)
            decoded_boxlist_list.append(decoded_boxlist)
        return self._hard_example_miner(location_losses=location_losses, cls_losses=cls_losses, decoded_boxlist_list=decoded_boxlist_list, match_list=match_list)

    def _batch_decode(self, box_encodings, anchors):
        if False:
            i = 10
            return i + 15
        'Decodes a batch of box encodings with respect to the anchors.\n\n    Args:\n      box_encodings: A float32 tensor of shape\n        [batch_size, num_anchors, box_code_size] containing box encodings.\n      anchors: A tensor of shape [num_anchors, 4].\n\n    Returns:\n      decoded_boxes: A float32 tensor of shape\n        [batch_size, num_anchors, 4] containing the decoded boxes.\n      decoded_keypoints: A float32 tensor of shape\n        [batch_size, num_anchors, num_keypoints, 2] containing the decoded\n        keypoints if present in the input `box_encodings`, None otherwise.\n    '
        combined_shape = shape_utils.combined_static_and_dynamic_shape(box_encodings)
        batch_size = combined_shape[0]
        tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
        tiled_anchors_boxlist = box_list.BoxList(tf.reshape(tiled_anchor_boxes, [-1, 4]))
        decoded_boxes = self._box_coder.decode(tf.reshape(box_encodings, [-1, self._box_coder.code_size]), tiled_anchors_boxlist)
        decoded_keypoints = None
        if decoded_boxes.has_field(fields.BoxListFields.keypoints):
            decoded_keypoints = decoded_boxes.get_field(fields.BoxListFields.keypoints)
            num_keypoints = decoded_keypoints.get_shape()[1]
            decoded_keypoints = tf.reshape(decoded_keypoints, tf.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
        decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack([combined_shape[0], combined_shape[1], 4]))
        return (decoded_boxes, decoded_keypoints)

    def regularization_losses(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of regularization losses for this model.\n\n    Returns a list of regularization losses for this model that the estimator\n    needs to use during training/optimization.\n\n    Returns:\n      A list of regularization loss tensors.\n    '
        losses = []
        slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if slim_losses:
            losses.extend(slim_losses)
        if self._box_predictor.is_keras_model:
            losses.extend(self._box_predictor.losses)
        if self._feature_extractor.is_keras_model:
            losses.extend(self._feature_extractor.losses)
        return losses

    def restore_map(self, fine_tune_checkpoint_type='detection', load_all_detection_checkpoint_vars=False):
        if False:
            return 10
        "Returns a map of variables to load from a foreign checkpoint.\n\n    See parent class for details.\n\n    Args:\n      fine_tune_checkpoint_type: whether to restore from a full detection\n        checkpoint (with compatible variable names) or to restore from a\n        classification checkpoint for initialization prior to training.\n        Valid values: `detection`, `classification`. Default 'detection'.\n      load_all_detection_checkpoint_vars: whether to load all variables (when\n         `fine_tune_checkpoint_type='detection'`). If False, only variables\n         within the appropriate scopes are included. Default False.\n\n    Returns:\n      A dict mapping variable names (to load from a checkpoint) to variables in\n      the model graph.\n    Raises:\n      ValueError: if fine_tune_checkpoint_type is neither `classification`\n        nor `detection`.\n    "
        if fine_tune_checkpoint_type == 'classification':
            return self._feature_extractor.restore_from_classification_checkpoint_fn(self._extract_features_scope)
        elif fine_tune_checkpoint_type == 'detection':
            variables_to_restore = {}
            if tf.executing_eagerly():
                if load_all_detection_checkpoint_vars:
                    for variable in self.variables:
                        var_name = variable.name[:-2]
                        variables_to_restore[var_name] = variable
                else:
                    for variable in self._feature_extractor.variables:
                        var_name = variable.name[:-2]
                        variables_to_restore[var_name] = variable
            else:
                for variable in variables_helper.get_global_variables_safely():
                    var_name = variable.op.name
                    if load_all_detection_checkpoint_vars:
                        variables_to_restore[var_name] = variable
                    elif var_name.startswith(self._extract_features_scope):
                        variables_to_restore[var_name] = variable
            return variables_to_restore
        else:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(fine_tune_checkpoint_type))

    def updates(self):
        if False:
            return 10
        "Returns a list of update operators for this model.\n\n    Returns a list of update operators for this model that must be executed at\n    each training step. The estimator's train op needs to have a control\n    dependency on these updates.\n\n    Returns:\n      A list of update operators.\n    "
        update_ops = []
        slim_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if slim_update_ops:
            update_ops.extend(slim_update_ops)
        if self._box_predictor.is_keras_model:
            update_ops.extend(self._box_predictor.get_updates_for(None))
            update_ops.extend(self._box_predictor.get_updates_for(self._box_predictor.inputs))
        if self._feature_extractor.is_keras_model:
            update_ops.extend(self._feature_extractor.get_updates_for(None))
            update_ops.extend(self._feature_extractor.get_updates_for(self._feature_extractor.inputs))
        return update_ops