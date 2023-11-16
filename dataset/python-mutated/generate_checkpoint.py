"""Standalone utility to generate some test saved models."""
from absl import app
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import dtypes
from tensorflow.python.module import module
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables

class TableModule(module.Module):
    """Three vars (one in a sub-module) and compute method."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        default_value = -1
        empty_key = 0
        deleted_key = -1
        self.lookup_table = lookup_ops.DenseHashTable(dtypes.int64, dtypes.int64, default_value=default_value, empty_key=empty_key, deleted_key=deleted_key, name='t1', initial_num_buckets=32)
        self.lookup_table.insert(1, 1)
        self.lookup_table.insert(2, 4)

class VariableModule(module.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.v = variables.Variable([1.0, 2.0, 3.0])
        self.w = variables.Variable([4.0, 5.0])
MODULE_CTORS = {'TableModule': TableModule, 'VariableModule': VariableModule}

def main(args):
    if False:
        print('Hello World!')
    if len(args) != 3:
        print('Expected: {export_path} {ModuleName}')
        print('Allowed ModuleNames:', MODULE_CTORS.keys())
        return 1
    (_, export_path, module_name) = args
    module_ctor = MODULE_CTORS.get(module_name)
    if not module_ctor:
        print('Expected ModuleName to be one of:', MODULE_CTORS.keys())
        return 2
    tf_module = module_ctor()
    ckpt = checkpoint.Checkpoint(tf_module)
    ckpt.write(export_path)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    app.run(main)