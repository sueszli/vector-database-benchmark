"""mlir is an experimental library that provides support APIs for MLIR."""
from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export

@tf_export('mlir.experimental.convert_graph_def')
def convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline', show_debug_info=False):
    if False:
        for i in range(10):
            print('nop')
    'Import a GraphDef and convert it to a textual MLIR module.\n\n  This API is only intended for inspecting the internals of TensorFlow and the\n  string returned is at the moment intended for debugging purposes.\n\n  Args:\n    graph_def: An object of type graph_pb2.GraphDef or a textual proto\n      representation of a valid GraphDef.\n    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the\n      module, see MLIR documentation for the [textual pass pipeline\n      syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).\n    show_debug_info: Whether to include locations in the emitted textual form.\n\n  Returns:\n    A textual representation of the MLIR module corresponding to the graphdef.\n\n  Raises:\n    InvalidArgumentError: if graph_def is invalid or cannot be converted to\n      MLIR.\n  '
    return pywrap_mlir.import_graphdef(graph_def, pass_pipeline, show_debug_info)

@tf_export('mlir.experimental.convert_function')
def convert_function(concrete_function, pass_pipeline='tf-standard-pipeline', show_debug_info=False):
    if False:
        while True:
            i = 10
    "Import a ConcreteFunction and convert it to a textual MLIR module.\n\n  This API is only intended for inspecting the internals of TensorFlow and the\n  string returned is at the moment intended for debugging purposes.\n\n  A [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) can be\n  imported and converted from TensorFlow to TensorFlow MLIR with this API by\n  extracting its ConcreteFunction (eagerly-executing wrapper around a\n  [tf.Graph](https://www.tensorflow.org/api_docs/python/tf/Graph)).\n\n  For example:\n  >>> @tf.function\n  ... def add(a, b):\n  ...   return a + b\n\n  >>> concrete_function = add.get_concrete_function(\n  ...     tf.TensorSpec(None, tf.dtypes.float32),\n  ...     tf.TensorSpec(None, tf.dtypes.float32))\n  >>> tf.mlir.experimental.convert_function(concrete_function)\n  '...module attributes {...} {...}...'\n\n  Args:\n    concrete_function: An object of type ConcreteFunction.\n    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the\n      module, see MLIR documentation for the [textual pass pipeline\n      syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).\n    show_debug_info: Whether to include locations in the emitted textual form.\n\n  Returns:\n    A textual representation of the MLIR module corresponding to the\n    ConcreteFunction.\n\n  Raises:\n    InvalidArgumentError: if concrete_function is invalid or cannot be converted\n      to MLIR.\n  "
    return pywrap_mlir.import_function(concrete_function, pass_pipeline, show_debug_info)

@tf_export('mlir.experimental.convert_saved_model')
def convert_saved_model(saved_model_path, exported_names, show_debug_info=False):
    if False:
        print('Hello World!')
    'Converts a SavedModel to MLIR module.\n\n  Args:\n    saved_model_path: Path to SavedModel.\n    exported_names: Names to export.\n    show_debug_info: Whether to include locations in the emitted textual form.\n\n  Returns:\n    A textual representation of the MLIR module corresponding to the\n    SavedModel.\n  '
    return pywrap_mlir.experimental_convert_saved_model_to_mlir(saved_model_path, exported_names, show_debug_info)

@tf_export('mlir.experimental.convert_saved_model_v1')
def convert_saved_model_v1(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy=True, show_debug_info=False):
    if False:
        print('Hello World!')
    'Converts a v1 SavedModel to MLIR module.\n\n  Args:\n    saved_model_path: Path to SavedModel.\n    exported_names: Names to export.\n    tags: MetaGraphDef to be loaded is identified by the supplied tags.\n    lift_variables: Whether to promote tf.VarHandleOp to resource arguments.\n    include_variables_in_initializers: Keeps the variables in initializers\n      before lifting variables.\n    upgrade_legacy: Functionalize the input graph before importing.\n    show_debug_info: Whether to include locations in the emitted textual form.\n\n  Returns:\n    A textual representation of the MLIR module corresponding to the\n    SavedModule.\n  '
    return pywrap_mlir.experimental_convert_saved_model_v1_to_mlir(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info)

@tf_export('mlir.experimental.run_pass_pipeline')
def run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info=False):
    if False:
        i = 10
        return i + 15
    'Runs a pipeline over input module.\n\n  Args:\n    mlir_txt: Textual representation of the MLIR module.\n    pass_pipeline: Pass pipeline to run on module.\n    show_debug_info: Whether to include locations in the emitted textual form.\n\n  Returns:\n    A textual representation of the MLIR module corresponding to the\n    transformed module.\n  '
    return pywrap_mlir.experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info)

@tf_export('mlir.experimental.write_bytecode')
def experimental_write_bytecode(filename, mlir_txt):
    if False:
        while True:
            i = 10
    'Writes an MLIR module out as bytecode.\n\n  Args:\n    filename: The filename to write to.\n    mlir_txt: The MLIR module in textual format.\n  '
    pywrap_mlir.experimental_write_bytecode(filename, mlir_txt)

@tf_export('mlir.experimental.tflite_to_tosa_bytecode')
def tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant=False, ordered_input_arrays=None, ordered_output_arrays=None):
    if False:
        while True:
            i = 10
    'Converts TFLite flatbuffer to TOSA dialect in MLIR bytecode.\n\n  Args:\n    flatbuffer: Path to flatbuffer.\n    bytecode: Path to output bytecode.\n    use_external_constant: Whether to create `tfl.external_const` instead of\n      `tfl.const`.\n    ordered_input_arrays:\n    ordered_output_arrays: If ordered_output_arrays is not empty, then the\n      function will only return nodes in ordered_output_arrays in the same order\n  '
    pywrap_mlir.experimental_tflite_to_tosa_bytecode(flatbuffer, bytecode, use_external_constant, ordered_input_arrays, ordered_output_arrays)