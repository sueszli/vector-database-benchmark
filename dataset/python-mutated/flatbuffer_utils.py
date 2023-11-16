"""Utility functions for FlatBuffers.

All functions that are commonly used to work with FlatBuffers.

Refer to the tensorflow lite flatbuffer schema here:
tensorflow/lite/schema/schema.fbs
"""
import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
_TFLITE_FILE_IDENTIFIER = b'TFL3'

def convert_bytearray_to_object(model_bytearray):
    if False:
        i = 10
        return i + 15
    'Converts a tflite model from a bytearray to an object for parsing.'
    model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
    return schema_fb.ModelT.InitFromObj(model_object)

def read_model(input_tflite_file):
    if False:
        for i in range(10):
            print('nop')
    'Reads a tflite model as a python object.\n\n  Args:\n    input_tflite_file: Full path name to the input tflite file\n\n  Raises:\n    RuntimeError: If input_tflite_file path is invalid.\n    IOError: If input_tflite_file cannot be opened.\n\n  Returns:\n    A python object corresponding to the input tflite file.\n  '
    if not gfile.Exists(input_tflite_file):
        raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
    with gfile.GFile(input_tflite_file, 'rb') as input_file_handle:
        model_bytearray = bytearray(input_file_handle.read())
    model = convert_bytearray_to_object(model_bytearray)
    if sys.byteorder == 'big':
        byte_swap_tflite_model_obj(model, 'little', 'big')
    return model

def read_model_with_mutable_tensors(input_tflite_file):
    if False:
        return 10
    'Reads a tflite model as a python object with mutable tensors.\n\n  Similar to read_model() with the addition that the returned object has\n  mutable tensors (read_model() returns an object with immutable tensors).\n\n  NOTE: This API only works for TFLite generated with\n  _experimental_use_buffer_offset=false\n\n  Args:\n    input_tflite_file: Full path name to the input tflite file\n\n  Raises:\n    RuntimeError: If input_tflite_file path is invalid.\n    IOError: If input_tflite_file cannot be opened.\n\n  Returns:\n    A mutable python object corresponding to the input tflite file.\n  '
    return copy.deepcopy(read_model(input_tflite_file))

def convert_object_to_bytearray(model_object, extra_buffer=b''):
    if False:
        return 10
    'Converts a tflite model from an object to a immutable bytearray.'
    builder = flatbuffers.Builder(1024)
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
    model_bytearray = bytes(builder.Output())
    model_bytearray = model_bytearray + extra_buffer
    return model_bytearray

def write_model(model_object, output_tflite_file):
    if False:
        print('Hello World!')
    'Writes the tflite model, a python object, into the output file.\n\n  NOTE: This API only works for TFLite generated with\n  _experimental_use_buffer_offset=false\n\n  Args:\n    model_object: A tflite model as a python object\n    output_tflite_file: Full path name to the output tflite file.\n\n  Raises:\n    IOError: If output_tflite_file path is invalid or cannot be opened.\n  '
    if sys.byteorder == 'big':
        model_object = copy.deepcopy(model_object)
        byte_swap_tflite_model_obj(model_object, 'big', 'little')
    model_bytearray = convert_object_to_bytearray(model_object)
    with gfile.GFile(output_tflite_file, 'wb') as output_file_handle:
        output_file_handle.write(model_bytearray)

def strip_strings(model):
    if False:
        print('Hello World!')
    'Strips all nonessential strings from the model to reduce model size.\n\n  We remove the following strings:\n  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)\n  1. Model description\n  2. SubGraph name\n  3. Tensor names\n  We retain OperatorCode custom_code and Metadata name.\n\n  Args:\n    model: The model from which to remove nonessential strings.\n  '
    model.description = None
    for subgraph in model.subgraphs:
        subgraph.name = None
        for tensor in subgraph.tensors:
            tensor.name = None
    model.signatureDefs = None

def type_to_name(tensor_type):
    if False:
        print('Hello World!')
    'Converts a numerical enum to a readable tensor type.'
    for (name, value) in schema_fb.TensorType.__dict__.items():
        if value == tensor_type:
            return name
    return None

def randomize_weights(model, random_seed=0, buffers_to_skip=None):
    if False:
        i = 10
        return i + 15
    'Randomize weights in a model.\n\n  Args:\n    model: The model in which to randomize weights.\n    random_seed: The input to the random number generator (default value is 0).\n    buffers_to_skip: The list of buffer indices to skip. The weights in these\n      buffers are left unmodified.\n  '
    random.seed(random_seed)
    buffers = model.buffers
    buffer_ids = range(1, len(buffers))
    if buffers_to_skip is not None:
        buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]
    buffer_types = {}
    for graph in model.subgraphs:
        for op in graph.operators:
            if op.inputs is None:
                break
            for input_idx in op.inputs:
                tensor = graph.tensors[input_idx]
                buffer_types[tensor.buffer] = type_to_name(tensor.type)
    for i in buffer_ids:
        buffer_i_data = buffers[i].data
        buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
        if buffer_i_size == 0:
            continue
        buffer_type = buffer_types.get(i, 'INT8')
        if buffer_type.startswith('FLOAT'):
            format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
            for offset in range(0, buffer_i_size, struct.calcsize(format_code)):
                value = random.uniform(-0.5, 0.5)
                struct.pack_into(format_code, buffer_i_data, offset, value)
        else:
            for j in range(buffer_i_size):
                buffer_i_data[j] = random.randint(0, 255)

def rename_custom_ops(model, map_custom_op_renames):
    if False:
        while True:
            i = 10
    'Rename custom ops so they use the same naming style as builtin ops.\n\n  Args:\n    model: The input tflite model.\n    map_custom_op_renames: A mapping from old to new custom op names.\n  '
    for op_code in model.operatorCodes:
        if op_code.customCode:
            op_code_str = op_code.customCode.decode('ascii')
            if op_code_str in map_custom_op_renames:
                op_code.customCode = map_custom_op_renames[op_code_str].encode('ascii')

def opcode_to_name(model, op_code):
    if False:
        return 10
    'Converts a TFLite op_code to the human readable name.\n\n  Args:\n    model: The input tflite model.\n    op_code: The op_code to resolve to a readable name.\n\n  Returns:\n    A string containing the human readable op name, or None if not resolvable.\n  '
    op = model.operatorCodes[op_code]
    code = max(op.builtinCode, op.deprecatedBuiltinCode)
    for (name, value) in vars(schema_fb.BuiltinOperator).items():
        if value == code:
            return name
    return None

def xxd_output_to_bytes(input_cc_file):
    if False:
        while True:
            i = 10
    'Converts xxd output C++ source file to bytes (immutable).\n\n  Args:\n    input_cc_file: Full path name to th C++ source file dumped by xxd\n\n  Raises:\n    RuntimeError: If input_cc_file path is invalid.\n    IOError: If input_cc_file cannot be opened.\n\n  Returns:\n    A bytearray corresponding to the input cc file array.\n  '
    pattern = re.compile('\\W*(0x[0-9a-fA-F,x ]+).*')
    model_bytearray = bytearray()
    with open(input_cc_file) as file_handle:
        for line in file_handle:
            values_match = pattern.match(line)
            if values_match is None:
                continue
            list_text = values_match.group(1)
            values_text = filter(None, list_text.split(','))
            values = [int(x, base=16) for x in values_text]
            model_bytearray.extend(values)
    return bytes(model_bytearray)

def xxd_output_to_object(input_cc_file):
    if False:
        i = 10
        return i + 15
    'Converts xxd output C++ source file to object.\n\n  Args:\n    input_cc_file: Full path name to th C++ source file dumped by xxd\n\n  Raises:\n    RuntimeError: If input_cc_file path is invalid.\n    IOError: If input_cc_file cannot be opened.\n\n  Returns:\n    A python object corresponding to the input tflite file.\n  '
    model_bytes = xxd_output_to_bytes(input_cc_file)
    return convert_bytearray_to_object(model_bytes)

def byte_swap_buffer_content(buffer, chunksize, from_endiness, to_endiness):
    if False:
        return 10
    'Helper function for byte-swapping the buffers field.'
    to_swap = [buffer.data[i:i + chunksize] for i in range(0, len(buffer.data), chunksize)]
    buffer.data = b''.join([int.from_bytes(byteswap, from_endiness).to_bytes(chunksize, to_endiness) for byteswap in to_swap])

def byte_swap_string_content(buffer, from_endiness, to_endiness):
    if False:
        while True:
            i = 10
    'Helper function for byte-swapping the string buffer.\n\n  Args:\n    buffer: TFLite string buffer of from_endiness format.\n    from_endiness: The original endianness format of the string buffer.\n    to_endiness: The destined endianness format of the string buffer.\n  '
    num_of_strings = int.from_bytes(buffer.data[0:4], from_endiness)
    string_content = bytearray(buffer.data[4 * (num_of_strings + 2):])
    prefix_data = b''.join([int.from_bytes(buffer.data[i:i + 4], from_endiness).to_bytes(4, to_endiness) for i in range(0, (num_of_strings + 1) * 4 + 1, 4)])
    buffer.data = prefix_data + string_content

def byte_swap_tflite_model_obj(model, from_endiness, to_endiness):
    if False:
        print('Hello World!')
    'Byte swaps the buffers field in a TFLite model.\n\n  Args:\n    model: TFLite model object of from_endiness format.\n    from_endiness: The original endianness format of the buffers in model.\n    to_endiness: The destined endianness format of the buffers in model.\n  '
    if model is None:
        return
    buffer_swapped = []
    types_of_16_bits = [schema_fb.TensorType.FLOAT16, schema_fb.TensorType.INT16, schema_fb.TensorType.UINT16]
    types_of_32_bits = [schema_fb.TensorType.FLOAT32, schema_fb.TensorType.INT32, schema_fb.TensorType.COMPLEX64, schema_fb.TensorType.UINT32]
    types_of_64_bits = [schema_fb.TensorType.INT64, schema_fb.TensorType.FLOAT64, schema_fb.TensorType.COMPLEX128, schema_fb.TensorType.UINT64]
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            if tensor.buffer > 0 and tensor.buffer < len(model.buffers) and (tensor.buffer not in buffer_swapped) and (model.buffers[tensor.buffer].data is not None):
                if tensor.type == schema_fb.TensorType.STRING:
                    byte_swap_string_content(model.buffers[tensor.buffer], from_endiness, to_endiness)
                elif tensor.type in types_of_16_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 2, from_endiness, to_endiness)
                elif tensor.type in types_of_32_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 4, from_endiness, to_endiness)
                elif tensor.type in types_of_64_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 8, from_endiness, to_endiness)
                else:
                    continue
                buffer_swapped.append(tensor.buffer)

def byte_swap_tflite_buffer(tflite_model, from_endiness, to_endiness):
    if False:
        return 10
    'Generates a new model byte array after byte swapping its buffers field.\n\n  Args:\n    tflite_model: TFLite flatbuffer in a byte array.\n    from_endiness: The original endianness format of the buffers in\n      tflite_model.\n    to_endiness: The destined endianness format of the buffers in tflite_model.\n\n  Returns:\n    TFLite flatbuffer in a byte array, after being byte swapped to to_endiness\n    format.\n  '
    if tflite_model is None:
        return None
    model = convert_bytearray_to_object(tflite_model)
    byte_swap_tflite_model_obj(model, from_endiness, to_endiness)
    return convert_object_to_bytearray(model)

def count_resource_variables(model):
    if False:
        return 10
    'Calculates the number of unique resource variables in a model.\n\n  Args:\n    model: the input tflite model, either as bytearray or object.\n\n  Returns:\n    An integer number representing the number of unique resource variables.\n  '
    if not isinstance(model, schema_fb.ModelT):
        model = convert_bytearray_to_object(model)
    unique_shared_names = set()
    for subgraph in model.subgraphs:
        if subgraph.operators is None:
            continue
        for op in subgraph.operators:
            builtin_code = schema_util.get_builtin_code_from_operator_code(model.operatorCodes[op.opcodeIndex])
            if builtin_code == schema_fb.BuiltinOperator.VAR_HANDLE:
                unique_shared_names.add(op.builtinOptions.sharedName)
    return len(unique_shared_names)