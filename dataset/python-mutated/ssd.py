"""Python library for ssd model, tailored for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
(major, minor, _) = tf.__version__.split('.')
if int(major) < 1 or (int(major == 1) and int(minor) < 14):
    raise RuntimeError('TensorFlow version >= 1.14 is required. Found ({}).'.format(tf.__version__))
from tensorflow.python.framework import function
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu.ops import tpu_ops
from object_detection import exporter
from object_detection.builders import model_builder
from object_detection.tpu_exporters import utils
ANCHORS = 'anchors'
BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'

def get_prediction_tensor_shapes(pipeline_config):
    if False:
        while True:
            i = 10
    "Gets static shapes of tensors by building the graph on CPU.\n\n  This function builds the graph on CPU and obtain static shapes of output\n  tensors from TPUPartitionedCall. Shapes information are later used for setting\n  shapes of tensors when TPU graphs are built. This is necessary because tensors\n  coming out of TPUPartitionedCall lose their shape information, which are\n  needed for a lot of CPU operations later.\n  Args:\n    pipeline_config: A TrainEvalPipelineConfig proto.\n\n  Returns:\n    A python dict of tensors' names and their shapes.\n  "
    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    (_, input_tensors) = exporter.input_placeholder_fn_map['image_tensor']()
    inputs = tf.cast(input_tensors, dtype=tf.float32)
    (preprocessed_inputs, true_image_shapes) = detection_model.preprocess(inputs)
    prediction_dict = detection_model.predict(preprocessed_inputs, true_image_shapes)
    return {BOX_ENCODINGS: prediction_dict[BOX_ENCODINGS].shape.as_list(), CLASS_PREDICTIONS_WITH_BACKGROUND: prediction_dict[CLASS_PREDICTIONS_WITH_BACKGROUND].shape.as_list(), ANCHORS: prediction_dict[ANCHORS].shape.as_list()}

def recover_shape(preprocessed_inputs, prediction_outputs, shapes_info):
    if False:
        for i in range(10):
            print('nop')
    'Recovers shape from TPUPartitionedCall.\n\n  Args:\n    preprocessed_inputs: 4D tensor, shaped (batch, channels, height, width)\n    prediction_outputs: Python list of tensors, in the following order -\n      box_encodings - 3D tensor, shaped (code_size, batch, num_anchors);\n      class_predictions_with_background - 3D tensor, shaped (num_classes + 1,\n      batch, num_anchors); anchors - 2D tensor, shaped (4, num_anchors)\n    shapes_info: Python dict of tensor shapes as lists.\n\n  Returns:\n    preprocessed_inputs: 4D tensor, shaped (batch, height, width, channels)\n    box_encodings: 3D tensor, shaped (batch, num_anchors, code_size)\n    class_predictions_with_background: 3D tensor,\n        shaped (batch, num_anchors, num_classes + 1)\n    anchors: 2D tensor, shaped (num_anchors, 4)\n  '
    preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 2, 3, 1])
    box_encodings = tf.transpose(prediction_outputs[0], perm=[1, 2, 0])
    box_encodings.set_shape(shapes_info[BOX_ENCODINGS])
    class_predictions_with_background = tf.transpose(prediction_outputs[1], perm=[1, 2, 0])
    class_predictions_with_background.set_shape(shapes_info[CLASS_PREDICTIONS_WITH_BACKGROUND])
    anchors = tf.transpose(prediction_outputs[2], perm=[1, 0])
    anchors.set_shape(shapes_info[ANCHORS])
    return (preprocessed_inputs, box_encodings, class_predictions_with_background, anchors)

def build_graph(pipeline_config, shapes_info, input_type='encoded_image_string_tensor', use_bfloat16=False):
    if False:
        i = 10
        return i + 15
    "Builds TPU serving graph of ssd to be exported.\n\n  Args:\n    pipeline_config: A TrainEvalPipelineConfig proto.\n    shapes_info: A python dict of tensors' names and their shapes, returned by\n      `get_prediction_tensor_shapes()`.\n    input_type: One of\n                'encoded_image_string_tensor': a 1d tensor with dtype=tf.string\n                'image_tensor': a 4d tensor with dtype=tf.uint8\n                'tf_example': a 1d tensor with dtype=tf.string\n    use_bfloat16: If true, use tf.bfloat16 on TPU.\n\n  Returns:\n    placeholder_tensor: A placeholder tensor, type determined by `input_type`.\n    result_tensor_dict: A python dict of tensors' names and tensors.\n  "
    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    (placeholder_tensor, input_tensors) = exporter.input_placeholder_fn_map[input_type]()
    inputs = tf.cast(input_tensors, dtype=tf.float32)
    (preprocessed_inputs, true_image_shapes) = detection_model.preprocess(inputs)
    preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 3, 1, 2])
    if use_bfloat16:
        preprocessed_inputs = tf.cast(preprocessed_inputs, dtype=tf.bfloat16)

    def predict_tpu_subgraph(preprocessed_inputs, true_image_shapes):
        if False:
            return 10
        "Wraps over the CPU version of `predict()`.\n\n    This builds a same graph as the original `predict()`, manipulates\n    result tensors' dimensions to be memory efficient on TPU, and\n    returns them as list of tensors.\n\n    Args:\n      preprocessed_inputs: A 4D tensor of shape (batch, channels, height, width)\n      true_image_shapes: True image shapes tensor.\n\n    Returns:\n      A Python list of tensors:\n        box_encodings: 3D tensor of shape (code_size, batch_size, num_anchors)\n        class_predictions_with_background: 3D tensor,\n            shape (num_classes + 1, batch_size, num_anchors)\n        anchors: 2D tensor of shape (4, num_anchors)\n    "
        preprocessed_inputs = tf.transpose(preprocessed_inputs, perm=[0, 2, 3, 1])
        if use_bfloat16:
            with tf.contrib.tpu.bfloat16_scope():
                prediction_dict = detection_model.predict(preprocessed_inputs, true_image_shapes)
        else:
            prediction_dict = detection_model.predict(preprocessed_inputs, true_image_shapes)
        return [tf.transpose(prediction_dict[BOX_ENCODINGS], perm=[2, 0, 1]), tf.transpose(prediction_dict[CLASS_PREDICTIONS_WITH_BACKGROUND], perm=[2, 0, 1]), tf.transpose(prediction_dict[ANCHORS], perm=[1, 0])]

    @function.Defun(capture_resource_var_by_value=False)
    def predict_tpu():
        if False:
            i = 10
            return i + 15
        return tf.contrib.tpu.rewrite(predict_tpu_subgraph, [preprocessed_inputs, true_image_shapes])
    prediction_outputs = tpu_functional.TPUPartitionedCall(args=predict_tpu.captured_inputs, device_ordinal=tpu_ops.tpu_ordinal_selector(), Tout=[o.type for o in predict_tpu.definition.signature.output_arg], f=predict_tpu)
    (preprocessed_inputs, box_encodings, class_predictions_with_background, anchors) = recover_shape(preprocessed_inputs, prediction_outputs, shapes_info)
    output_tensors = {'preprocessed_inputs': preprocessed_inputs, BOX_ENCODINGS: box_encodings, CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_with_background, ANCHORS: anchors}
    if use_bfloat16:
        output_tensors = utils.bfloat16_to_float32_nested(output_tensors)
    postprocessed_tensors = detection_model.postprocess(output_tensors, true_image_shapes)
    result_tensor_dict = exporter.add_output_tensor_nodes(postprocessed_tensors, 'inference_op')
    return (placeholder_tensor, result_tensor_dict)