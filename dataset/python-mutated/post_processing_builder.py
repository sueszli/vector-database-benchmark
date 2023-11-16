"""Builder function for post processing operations."""
import functools
import tensorflow as tf
from object_detection.builders import calibration_builder
from object_detection.core import post_processing
from object_detection.protos import post_processing_pb2

def build(post_processing_config):
    if False:
        i = 10
        return i + 15
    'Builds callables for post-processing operations.\n\n  Builds callables for non-max suppression, score conversion, and (optionally)\n  calibration based on the configuration.\n\n  Non-max suppression callable takes `boxes`, `scores`, and optionally\n  `clip_window`, `parallel_iterations` `masks, and `scope` as inputs. It returns\n  `nms_boxes`, `nms_scores`, `nms_classes` `nms_masks` and `num_detections`. See\n  post_processing.batch_multiclass_non_max_suppression for the type and shape\n  of these tensors.\n\n  Score converter callable should be called with `input` tensor. The callable\n  returns the output from one of 3 tf operations based on the configuration -\n  tf.identity, tf.sigmoid or tf.nn.softmax. If a calibration config is provided,\n  score_converter also applies calibration transformations, as defined in\n  calibration_builder.py. See tensorflow documentation for argument and return\n  value descriptions.\n\n  Args:\n    post_processing_config: post_processing.proto object containing the\n      parameters for the post-processing operations.\n\n  Returns:\n    non_max_suppressor_fn: Callable for non-max suppression.\n    score_converter_fn: Callable for score conversion.\n\n  Raises:\n    ValueError: if the post_processing_config is of incorrect type.\n  '
    if not isinstance(post_processing_config, post_processing_pb2.PostProcessing):
        raise ValueError('post_processing_config not of type post_processing_pb2.Postprocessing.')
    non_max_suppressor_fn = _build_non_max_suppressor(post_processing_config.batch_non_max_suppression)
    score_converter_fn = _build_score_converter(post_processing_config.score_converter, post_processing_config.logit_scale)
    if post_processing_config.HasField('calibration_config'):
        score_converter_fn = _build_calibrated_score_converter(score_converter_fn, post_processing_config.calibration_config)
    return (non_max_suppressor_fn, score_converter_fn)

def _build_non_max_suppressor(nms_config):
    if False:
        while True:
            i = 10
    'Builds non-max suppresson based on the nms config.\n\n  Args:\n    nms_config: post_processing_pb2.PostProcessing.BatchNonMaxSuppression proto.\n\n  Returns:\n    non_max_suppressor_fn: Callable non-max suppressor.\n\n  Raises:\n    ValueError: On incorrect iou_threshold or on incompatible values of\n      max_total_detections and max_detections_per_class or on negative\n      soft_nms_sigma.\n  '
    if nms_config.iou_threshold < 0 or nms_config.iou_threshold > 1.0:
        raise ValueError('iou_threshold not in [0, 1.0].')
    if nms_config.max_detections_per_class > nms_config.max_total_detections:
        raise ValueError('max_detections_per_class should be no greater than max_total_detections.')
    if nms_config.soft_nms_sigma < 0.0:
        raise ValueError('soft_nms_sigma should be non-negative.')
    if nms_config.use_combined_nms and nms_config.use_class_agnostic_nms:
        raise ValueError('combined_nms does not support class_agnostic_nms.')
    non_max_suppressor_fn = functools.partial(post_processing.batch_multiclass_non_max_suppression, score_thresh=nms_config.score_threshold, iou_thresh=nms_config.iou_threshold, max_size_per_class=nms_config.max_detections_per_class, max_total_size=nms_config.max_total_detections, use_static_shapes=nms_config.use_static_shapes, use_class_agnostic_nms=nms_config.use_class_agnostic_nms, max_classes_per_detection=nms_config.max_classes_per_detection, soft_nms_sigma=nms_config.soft_nms_sigma, use_partitioned_nms=nms_config.use_partitioned_nms, use_combined_nms=nms_config.use_combined_nms, change_coordinate_frame=True)
    return non_max_suppressor_fn

def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
    if False:
        i = 10
        return i + 15
    'Create a function to scale logits then apply a Tensorflow function.'

    def score_converter_fn(logits):
        if False:
            i = 10
            return i + 15
        scaled_logits = tf.divide(logits, logit_scale, name='scale_logits')
        return tf_score_converter_fn(scaled_logits, name='convert_scores')
    score_converter_fn.__name__ = '%s_with_logit_scale' % tf_score_converter_fn.__name__
    return score_converter_fn

def _build_score_converter(score_converter_config, logit_scale):
    if False:
        while True:
            i = 10
    'Builds score converter based on the config.\n\n  Builds one of [tf.identity, tf.sigmoid, tf.softmax] score converters based on\n  the config.\n\n  Args:\n    score_converter_config: post_processing_pb2.PostProcessing.score_converter.\n    logit_scale: temperature to use for SOFTMAX score_converter.\n\n  Returns:\n    Callable score converter op.\n\n  Raises:\n    ValueError: On unknown score converter.\n  '
    if score_converter_config == post_processing_pb2.PostProcessing.IDENTITY:
        return _score_converter_fn_with_logit_scale(tf.identity, logit_scale)
    if score_converter_config == post_processing_pb2.PostProcessing.SIGMOID:
        return _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale)
    if score_converter_config == post_processing_pb2.PostProcessing.SOFTMAX:
        return _score_converter_fn_with_logit_scale(tf.nn.softmax, logit_scale)
    raise ValueError('Unknown score converter.')

def _build_calibrated_score_converter(score_converter_fn, calibration_config):
    if False:
        return 10
    "Wraps a score_converter_fn, adding a calibration step.\n\n  Builds a score converter function with a calibration transformation according\n  to calibration_builder.py. The score conversion function may be applied before\n  or after the calibration transformation, depending on the calibration method.\n  If the method is temperature scaling, the score conversion is\n  after the calibration transformation. Otherwise, the score conversion is\n  before the calibration transformation. Calibration applies positive monotonic\n  transformations to inputs (i.e. score ordering is strictly preserved or\n  adjacent scores are mapped to the same score). When calibration is\n  class-agnostic, the highest-scoring class remains unchanged, unless two\n  adjacent scores are mapped to the same value and one class arbitrarily\n  selected to break the tie. In per-class calibration, it's possible (though\n  rare in practice) that the highest-scoring class will change, since positive\n  monotonicity is only required to hold within each class.\n\n  Args:\n    score_converter_fn: callable that takes logit scores as input.\n    calibration_config: post_processing_pb2.PostProcessing.calibration_config.\n\n  Returns:\n    Callable calibrated score coverter op.\n  "
    calibration_fn = calibration_builder.build(calibration_config)

    def calibrated_score_converter_fn(logits):
        if False:
            for i in range(10):
                print('nop')
        if calibration_config.WhichOneof('calibrator') == 'temperature_scaling_calibration':
            calibrated_logits = calibration_fn(logits)
            return score_converter_fn(calibrated_logits)
        else:
            converted_logits = score_converter_fn(logits)
            return calibration_fn(converted_logits)
    calibrated_score_converter_fn.__name__ = 'calibrate_with_%s' % calibration_config.WhichOneof('calibrator')
    return calibrated_score_converter_fn