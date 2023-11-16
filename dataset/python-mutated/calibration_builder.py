"""Tensorflow ops to calibrate class predictions and background class."""
import tensorflow as tf
from object_detection.utils import shape_utils

def _find_interval_containing_new_value(x, new_value):
    if False:
        print('Hello World!')
    'Find the index of x (ascending-ordered) after which new_value occurs.'
    new_value_shape = shape_utils.combined_static_and_dynamic_shape(new_value)[0]
    x_shape = shape_utils.combined_static_and_dynamic_shape(x)[0]
    compare = tf.cast(tf.reshape(new_value, shape=(new_value_shape, 1)) >= tf.reshape(x, shape=(1, x_shape)), dtype=tf.int32)
    diff = compare[:, 1:] - compare[:, :-1]
    interval_idx = tf.argmin(diff, axis=1)
    return interval_idx

def _tf_linear_interp1d(x_to_interpolate, fn_x, fn_y):
    if False:
        while True:
            i = 10
    'Tensorflow implementation of 1d linear interpolation.\n\n  Args:\n    x_to_interpolate: tf.float32 Tensor of shape (num_examples,) over which 1d\n      linear interpolation is performed.\n    fn_x: Monotonically-increasing, non-repeating tf.float32 Tensor of shape\n      (length,) used as the domain to approximate a function.\n    fn_y: tf.float32 Tensor of shape (length,) used as the range to approximate\n      a function.\n\n  Returns:\n    tf.float32 Tensor of shape (num_examples,)\n  '
    x_pad = tf.concat([fn_x[:1] - 1, fn_x, fn_x[-1:] + 1], axis=0)
    y_pad = tf.concat([fn_y[:1], fn_y, fn_y[-1:]], axis=0)
    interval_idx = _find_interval_containing_new_value(x_pad, x_to_interpolate)
    alpha = (x_to_interpolate - tf.gather(x_pad, interval_idx)) / (tf.gather(x_pad, interval_idx + 1) - tf.gather(x_pad, interval_idx))
    interpolation = (1 - alpha) * tf.gather(y_pad, interval_idx) + alpha * tf.gather(y_pad, interval_idx + 1)
    return interpolation

def _function_approximation_proto_to_tf_tensors(x_y_pairs_message):
    if False:
        for i in range(10):
            print('nop')
    'Extracts (x,y) pairs from a XYPairs message.\n\n  Args:\n    x_y_pairs_message: calibration_pb2..XYPairs proto\n  Returns:\n    tf_x: tf.float32 tensor of shape (number_xy_pairs,) for function domain.\n    tf_y: tf.float32 tensor of shape (number_xy_pairs,) for function range.\n  '
    tf_x = tf.convert_to_tensor([x_y_pair.x for x_y_pair in x_y_pairs_message.x_y_pair], dtype=tf.float32)
    tf_y = tf.convert_to_tensor([x_y_pair.y for x_y_pair in x_y_pairs_message.x_y_pair], dtype=tf.float32)
    return (tf_x, tf_y)

def _get_class_id_function_dict(calibration_config):
    if False:
        for i in range(10):
            print('nop')
    'Create a dictionary mapping class id to function approximations.\n\n  Args:\n    calibration_config: calibration_pb2 proto containing\n      id_function_approximations.\n  Returns:\n    Dictionary mapping a class id to a tuple of TF tensors to be used for\n    function approximation.\n  '
    class_id_function_dict = {}
    class_id_xy_pairs_map = calibration_config.class_id_function_approximations.class_id_xy_pairs_map
    for class_id in class_id_xy_pairs_map:
        class_id_function_dict[class_id] = _function_approximation_proto_to_tf_tensors(class_id_xy_pairs_map[class_id])
    return class_id_function_dict

def build(calibration_config):
    if False:
        while True:
            i = 10
    'Returns a function that calibrates Tensorflow model scores.\n\n  All returned functions are expected to apply positive monotonic\n  transformations to inputs (i.e. score ordering is strictly preserved or\n  adjacent scores are mapped to the same score, but an input of lower value\n  should never be exceed an input of higher value after transformation).  For\n  class-agnostic calibration, positive monotonicity should hold across all\n  scores. In class-specific cases, positive monotonicity should hold within each\n  class.\n\n  Args:\n    calibration_config: calibration_pb2.CalibrationConfig proto.\n  Returns:\n    Function that that accepts class_predictions_with_background and calibrates\n    the output based on calibration_config\'s parameters.\n  Raises:\n    ValueError: No calibration builder defined for "Oneof" in\n      calibration_config.\n  '
    if calibration_config.WhichOneof('calibrator') == 'function_approximation':

        def calibration_fn(class_predictions_with_background):
            if False:
                for i in range(10):
                    print('nop')
            'Calibrate predictions via 1-d linear interpolation.\n\n      Predictions scores are linearly interpolated based on a class-agnostic\n      function approximation. Note that the 0-indexed background class is also\n      transformed.\n\n      Args:\n        class_predictions_with_background: tf.float32 tensor of shape\n          [batch_size, num_anchors, num_classes + 1] containing scores on the\n          interval [0,1]. This is usually produced by a sigmoid or softmax layer\n          and the result of calling the `predict` method of a detection model.\n\n      Returns:\n        tf.float32 tensor of the same shape as the input with values on the\n        interval [0, 1].\n      '
            flat_class_predictions_with_background = tf.reshape(class_predictions_with_background, shape=[-1])
            (fn_x, fn_y) = _function_approximation_proto_to_tf_tensors(calibration_config.function_approximation.x_y_pairs)
            updated_scores = _tf_linear_interp1d(flat_class_predictions_with_background, fn_x, fn_y)
            original_detections_shape = shape_utils.combined_static_and_dynamic_shape(class_predictions_with_background)
            calibrated_class_predictions_with_background = tf.reshape(updated_scores, shape=original_detections_shape, name='calibrate_scores')
            return calibrated_class_predictions_with_background
    elif calibration_config.WhichOneof('calibrator') == 'class_id_function_approximations':

        def calibration_fn(class_predictions_with_background):
            if False:
                i = 10
                return i + 15
            "Calibrate predictions per class via 1-d linear interpolation.\n\n      Prediction scores are linearly interpolated with class-specific function\n      approximations. Note that after calibration, an anchor's class scores will\n      not necessarily sum to 1, and score ordering may change, depending on each\n      class' calibration parameters.\n\n      Args:\n        class_predictions_with_background: tf.float32 tensor of shape\n          [batch_size, num_anchors, num_classes + 1] containing scores on the\n          interval [0,1]. This is usually produced by a sigmoid or softmax layer\n          and the result of calling the `predict` method of a detection model.\n\n      Returns:\n        tf.float32 tensor of the same shape as the input with values on the\n        interval [0, 1].\n\n      Raises:\n        KeyError: Calibration parameters are not present for a class.\n      "
            class_id_function_dict = _get_class_id_function_dict(calibration_config)
            class_tensors = tf.unstack(class_predictions_with_background, axis=-1)
            calibrated_class_tensors = []
            for (class_id, class_tensor) in enumerate(class_tensors):
                flat_class_tensor = tf.reshape(class_tensor, shape=[-1])
                if class_id in class_id_function_dict:
                    output_tensor = _tf_linear_interp1d(x_to_interpolate=flat_class_tensor, fn_x=class_id_function_dict[class_id][0], fn_y=class_id_function_dict[class_id][1])
                else:
                    tf.logging.info('Calibration parameters for class id `%d` not not found', class_id)
                    output_tensor = flat_class_tensor
                calibrated_class_tensors.append(output_tensor)
            combined_calibrated_tensor = tf.stack(calibrated_class_tensors, axis=1)
            input_shape = shape_utils.combined_static_and_dynamic_shape(class_predictions_with_background)
            calibrated_class_predictions_with_background = tf.reshape(combined_calibrated_tensor, shape=input_shape, name='calibrate_scores')
            return calibrated_class_predictions_with_background
    elif calibration_config.WhichOneof('calibrator') == 'temperature_scaling_calibration':

        def calibration_fn(class_predictions_with_background):
            if False:
                return 10
            'Calibrate predictions via temperature scaling.\n\n      Predictions logits scores are scaled by the temperature scaler. Note that\n      the 0-indexed background class is also transformed.\n\n      Args:\n        class_predictions_with_background: tf.float32 tensor of shape\n          [batch_size, num_anchors, num_classes + 1] containing logits scores.\n          This is usually produced before a sigmoid or softmax layer.\n\n      Returns:\n        tf.float32 tensor of the same shape as the input.\n\n      Raises:\n        ValueError: If temperature scaler is of incorrect value.\n      '
            scaler = calibration_config.temperature_scaling_calibration.scaler
            if scaler <= 0:
                raise ValueError('The scaler in temperature scaling must be positive.')
            calibrated_class_predictions_with_background = tf.math.divide(class_predictions_with_background, scaler, name='calibrate_score')
            return calibrated_class_predictions_with_background
    else:
        raise ValueError('No calibration builder defined for "Oneof" in calibration_config.')
    return calibration_fn