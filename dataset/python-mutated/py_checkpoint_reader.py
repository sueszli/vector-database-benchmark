"""Extending CheckpointReader for TensorFlow."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export

def error_translator(e):
    if False:
        i = 10
        return i + 15
    'Translate the tensor_slice_reader.cc errors.'
    error_message = str(e)
    if 'not found in checkpoint' in error_message or 'Failed to find any matching files for' in error_message:
        raise errors_impl.NotFoundError(None, None, error_message)
    elif 'Sliced checkpoints are not supported' in error_message or 'Data type not supported' in error_message:
        raise errors_impl.UnimplementedError(None, None, error_message)
    elif 'Failed to get matching files on' in error_message:
        raise errors_impl.InvalidArgumentError(None, None, error_message)
    elif 'Unable to open table file' in error_message:
        raise errors_impl.DataLossError(None, None, error_message)
    elif 'Failed to find the saved tensor slices' in error_message or 'not convertible to numpy dtype' in error_message:
        raise errors_impl.InternalError(None, None, error_message)
    else:
        raise errors_impl.OpError(None, None, error_message, errors_impl.UNKNOWN)

def get_variable_to_dtype_map(self):
    if False:
        for i in range(10):
            print('nop')
    return {name: dtypes.DType(type_enum) for (name, type_enum) in self._GetVariableToDataTypeMap().items()}
CheckpointReader.get_variable_to_dtype_map = get_variable_to_dtype_map

def has_tensor(self, tensor_str):
    if False:
        for i in range(10):
            print('nop')
    return self._HasTensor(compat.as_bytes(tensor_str))
CheckpointReader.has_tensor = has_tensor

def get_tensor(self, tensor_str):
    if False:
        print('Hello World!')
    'Get the tensor from the Checkpoint object.'
    try:
        return CheckpointReader.CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
    except RuntimeError as e:
        error_translator(e)
CheckpointReader.get_tensor = get_tensor

@tf_export(v1=['train.NewCheckpointReader'])
def NewCheckpointReader(filepattern):
    if False:
        for i in range(10):
            print('nop')
    'A function that returns a CheckPointReader.\n\n  Args:\n    filepattern: The filename.\n\n  Returns:\n    A CheckpointReader object.\n  '
    try:
        return CheckpointReader(compat.as_bytes(filepattern))
    except RuntimeError as e:
        error_translator(e)