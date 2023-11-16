"""Standalone utility to generate some test saved models."""
import os
from absl import app
from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import asset

class VarsAndArithmeticObjectGraph(module.Module):
    """Three vars (one in a sub-module) and compute method."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = variables.Variable(1.0, name='variable_x')
        self.y = variables.Variable(2.0, name='variable_y')
        self.child = module.Module()
        self.child.z = variables.Variable(3.0, name='child_variable')
        self.child.c = ops.convert_to_tensor(5.0)

    @def_function.function(input_signature=[tensor_spec.TensorSpec((), dtypes.float32), tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, a, b):
        if False:
            return 10
        return (a + self.x) * (b + self.y) / self.child.z + self.child.c

class ReferencesParent(module.Module):

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super(ReferencesParent, self).__init__()
        self.parent = parent
        self.my_variable = variables.Variable(3.0, name='MyVariable')

class CyclicModule(module.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CyclicModule, self).__init__()
        self.child = ReferencesParent(self)

class AssetModule(module.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        self.asset = asset.Asset(test.test_src_dir_path('cc/saved_model/testdata/test_asset.txt'))

    @def_function.function(input_signature=[])
    def read_file(self):
        if False:
            for i in range(10):
                print('nop')
        return io_ops.read_file(self.asset)

class StaticHashTableModule(module.Module):
    """A module with an Asset, StaticHashTable, and a lookup function."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.asset = asset.Asset(test.test_src_dir_path('cc/saved_model/testdata/static_hashtable_asset.txt'))
        self.table = lookup_ops.StaticHashTable(lookup_ops.TextFileInitializer(self.asset, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE, dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER), -1)

    @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=None, dtype=dtypes.string)])
    def lookup(self, word):
        if False:
            print('Hello World!')
        return self.table.lookup(word)

def get_simple_session():
    if False:
        return 10
    ops.disable_eager_execution()
    sess = session_lib.Session()
    variables.Variable(1.0)
    sess.run(variables.global_variables_initializer())
    return sess
MODULE_CTORS = {'VarsAndArithmeticObjectGraph': (VarsAndArithmeticObjectGraph, 2), 'CyclicModule': (CyclicModule, 2), 'AssetModule': (AssetModule, 2), 'StaticHashTableModule': (StaticHashTableModule, 2), 'SimpleV1Model': (get_simple_session, 1)}

def main(args):
    if False:
        while True:
            i = 10
    if len(args) != 3:
        print('Expected: {export_path} {ModuleName}')
        print('Allowed ModuleNames:', MODULE_CTORS.keys())
        return 1
    (_, export_path, module_name) = args
    (module_ctor, version) = MODULE_CTORS.get(module_name)
    if not module_ctor:
        print('Expected ModuleName to be one of:', MODULE_CTORS.keys())
        return 2
    os.makedirs(export_path)
    tf_module = module_ctor()
    if version == 2:
        options = save_options.SaveOptions(save_debug_info=True)
        saved_model.save(tf_module, export_path, options=options)
    else:
        builder = saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(tf_module, ['serve'])
        builder.save()
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    app.run(main)