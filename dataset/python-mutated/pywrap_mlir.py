"""Python module for MLIR functions exported by pybind11."""
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *

def import_graphdef(graphdef, pass_pipeline, show_debug_info, input_names=None, input_data_types=None, input_data_shapes=None, output_names=[]):
    if False:
        for i in range(10):
            print('nop')
    if input_names is not None:
        return ImportGraphDef(str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info, ','.join(input_names).encode('utf-8'), ','.join(input_data_types).encode('utf-8'), ':'.join(input_data_shapes).encode('utf-8'), ','.join(output_names).encode('utf-8'))
    return ImportGraphDef(str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)

def import_function(concrete_function, pass_pipeline, show_debug_info):
    if False:
        return 10
    ctxt = context.context()
    ctxt.ensure_initialized()
    return ImportFunction(ctxt._handle, str(concrete_function.function_def).encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)

def experimental_convert_saved_model_to_mlir(saved_model_path, exported_names, show_debug_info):
    if False:
        print('Hello World!')
    return ExperimentalConvertSavedModelToMlir(str(saved_model_path).encode('utf-8'), str(exported_names).encode('utf-8'), show_debug_info)

def experimental_convert_saved_model_v1_to_mlir_lite(saved_model_path, exported_names, tags, upgrade_legacy, show_debug_info):
    if False:
        return 10
    return ExperimentalConvertSavedModelV1ToMlirLite(str(saved_model_path).encode('utf-8'), str(exported_names).encode('utf-8'), str(tags).encode('utf-8'), upgrade_legacy, show_debug_info)

def experimental_convert_saved_model_v1_to_mlir(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info):
    if False:
        print('Hello World!')
    return ExperimentalConvertSavedModelV1ToMlir(str(saved_model_path).encode('utf-8'), str(exported_names).encode('utf-8'), str(tags).encode('utf-8'), lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info)

def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
    if False:
        for i in range(10):
            print('nop')
    return ExperimentalRunPassPipeline(mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)

def experimental_write_bytecode(filename, mlir_txt):
    if False:
        for i in range(10):
            print('nop')
    return ExperimentalWriteBytecode(filename.encode('utf-8'), mlir_txt.encode())

def experimental_tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant=False, ordered_input_arrays=None, ordered_output_arrays=None):
    if False:
        print('Hello World!')
    if ordered_input_arrays is None:
        ordered_input_arrays = []
    if ordered_output_arrays is None:
        ordered_output_arrays = []
    return ExperimentalTFLiteToTosaBytecode(flatbuffer.encode('utf-8'), bytecode.encode('utf-8'), use_external_constant, ordered_input_arrays, ordered_output_arrays)