"""Converts a model's graph def into a tflite model with MLIR-based conversion."""
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.testing import zip_test_utils
from tensorflow.python.platform import resource_loader
from tensorflow.python.saved_model import signature_constants

def mlir_convert(options, saved_model_dir, input_tensors, output_tensors, **kwargs):
    if False:
        return 10
    'Convert a saved model into a tflite model with MLIR-based conversion.\n\n  Args:\n    options: A lite.testing.generate_examples_lib.Options instance.\n    saved_model_dir: Path to the saved model.\n    input_tensors: List of input tensor tuples `(name, shape, type)`.\n    output_tensors: List of output tensors (names).\n    **kwargs: Extra parameters.\n\n  Returns:\n    output tflite model, log_txt from conversion\n    or None, log_txt if it did not convert properly.\n  '
    test_params = kwargs.get('test_params', {})
    extra_convert_options = kwargs.get('extra_convert_options', zip_test_utils.ExtraConvertOptions())
    tflite_model = None
    log = ''
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, [signature_key])
    converter.allow_custom_ops = extra_convert_options.allow_custom_ops
    converter.experimental_new_quantizer = options.mlir_quantizer
    if options.make_tf_ptq_tests:
        if options.hlo_aware_conversion:
            tf_quantization_mode = 'DEFAULT'
        else:
            tf_quantization_mode = 'LEGACY_INTEGER'
        converter._experimental_tf_quantization_mode = tf_quantization_mode
    if options.run_with_flex:
        converter.target_spec.supported_ops = set([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
    if options.enable_dynamic_update_slice:
        converter._experimental_enable_dynamic_update_slice = True
    converter.unfold_batchmatmul = options.unfold_batchmatmul
    if test_params.get('dynamic_range_quantize', False):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if options.experimental_low_bit_qat:
        converter._experimental_low_bit_qat = True
    if test_params.get('fully_quantize', False):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        (min_value, max_value) = test_params.get('input_range', (-1, 1))

        def representative_dataset(input_tensors):
            if False:
                i = 10
                return i + 15
            calibration_inputs = {}
            for (name, shape, dtype) in input_tensors:
                if shape:
                    dims = [1 if dim.value is None else dim.value for dim in shape.dims]
                    calibration_inputs[name] = np.random.uniform(min_value, max_value, tuple(dims)).astype(dtype.as_numpy_dtype)
            return calibration_inputs

        def representative_dataset_gen():
            if False:
                print('Hello World!')
            for _ in range(100):
                yield representative_dataset(input_tensors)
        if test_params.get('quant_16x8', False):
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset_gen
        if extra_convert_options.inference_input_type:
            converter.inference_input_type = extra_convert_options.inference_input_type
        if extra_convert_options.inference_output_type:
            converter.inference_output_type = extra_convert_options.inference_output_type
    try:
        tflite_model = converter.convert()
        if options.expected_ops_in_converted_model:
            ops_list = tflite_test_util.get_ops_list(tflite_model)
            for expected_op in options.expected_ops_in_converted_model:
                if expected_op not in ops_list:
                    tflite_model = None
                    raise ValueError('{} op not found in the converted model'.format(expected_op))
    except Exception as e:
        log = str(e)
    return (tflite_model, log)

def mlir_convert_file(graph_def_filename, input_tensors, output_tensors, quantization_params=None, additional_flags=''):
    if False:
        for i in range(10):
            print('nop')
    "Convert a graphdef file into a tflite model with MLIR-based conversion.\n\n  NOTE: this currently shells out to the MLIR binary binary, but we would like\n  convert to Python API tooling in the future.\n\n  Args:\n    graph_def_filename: A GraphDef file.\n    input_tensors: List of input tensor tuples `(name, shape, type)`. name\n      should be a string. shape should be a tuple of integers. type should be a\n      string, for example 'DT_FLOAT'\n    output_tensors: List of output tensors (names).\n    quantization_params: parameters `(inference_type, min_values, max_values)`\n      to quantize the model.\n    additional_flags: A string of additional command line flags to be passed to\n      MLIR converter.\n\n  Returns:\n    output tflite model, log_txt from conversion\n    or None, log_txt if it did not convert properly.\n  "
    bin_path = resource_loader.get_path_to_datafile('../../../../compiler/mlir/lite/tf_tfl_translate')
    with tempfile.NamedTemporaryFile() as output_file, tempfile.NamedTemporaryFile('w+') as stdout_file:
        input_shapes = []
        for input_tensor in input_tensors:
            shape = input_tensor[1]
            input_shapes.append(','.join([str(dim) for dim in shape]))
        input_shapes_str = ':'.join(input_shapes)
        input_types = ','.join([x[2] for x in input_tensors])
        quant_flags = ''
        if quantization_params is not None:
            min_vals = ','.join([str(val) for val in quantization_params[1]])
            max_vals = ','.join([str(val) for val in quantization_params[2]])
            quant_flags = '-tf-inference-type=' + quantization_params[0] + " -tf-input-min-values='" + min_vals + "' -tf-input-max-values='" + max_vals + "' " + '-emit-quant-adaptor-ops '
        cmd = '%s -tf-input-arrays=%s -tf-input-data-types=%s -tf-input-shapes=%s -tf-output-arrays=%s ' + quant_flags + additional_flags + '%s -o %s'
        cmd = cmd % (bin_path, ','.join([x[0] for x in input_tensors]), input_types, input_shapes_str, ','.join(output_tensors), graph_def_filename, output_file.name)
        exit_code = os.system(cmd)
        log = cmd + 'exited with code %d' % exit_code + '\n------------------\n' + stdout_file.read()
        return (None if exit_code != 0 else output_file.read(), log)