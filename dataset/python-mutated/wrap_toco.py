"""Wraps toco interface with python lazy loader."""
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import _pywrap_toco_api

def wrapped_toco_convert(model_flags_str, toco_flags_str, input_data_str, debug_info_str, enable_mlir_converter):
    if False:
        return 10
    'Wraps TocoConvert with lazy loader.'
    return _pywrap_toco_api.TocoConvert(model_flags_str, toco_flags_str, input_data_str, False, debug_info_str, enable_mlir_converter)

def wrapped_experimental_mlir_quantize(input_data_str, disable_per_channel, fully_quantize, inference_type, input_data_type, output_data_type, enable_numeric_verify, enable_whole_model_verify, denylisted_ops, denylisted_nodes, enable_variable_quantization):
    if False:
        i = 10
        return i + 15
    'Wraps experimental mlir quantize model.'
    return _pywrap_toco_api.ExperimentalMlirQuantizeModel(input_data_str, disable_per_channel, fully_quantize, inference_type, input_data_type, output_data_type, enable_numeric_verify, enable_whole_model_verify, denylisted_ops, denylisted_nodes, enable_variable_quantization)

def wrapped_experimental_mlir_sparsify(input_data_str):
    if False:
        return 10
    'Wraps experimental mlir sparsify model.'
    return _pywrap_toco_api.ExperimentalMlirSparsifyModel(input_data_str)

def wrapped_register_custom_opdefs(custom_opdefs_list):
    if False:
        for i in range(10):
            print('nop')
    'Wraps RegisterCustomOpdefs with lazy loader.'
    return _pywrap_toco_api.RegisterCustomOpdefs(custom_opdefs_list)

def wrapped_retrieve_collected_errors():
    if False:
        return 10
    'Wraps RetrieveCollectedErrors with lazy loader.'
    return _pywrap_toco_api.RetrieveCollectedErrors()

def wrapped_flat_buffer_file_to_mlir(model, input_is_filepath):
    if False:
        for i in range(10):
            print('nop')
    'Wraps FlatBufferFileToMlir with lazy loader.'
    return _pywrap_toco_api.FlatBufferToMlir(model, input_is_filepath)