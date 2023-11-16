"""Utility classes for testing checkpointing."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module

class CheckpointedOp:
    """Op with a custom checkpointing implementation.

  Defined as part of the test because the MutableHashTable Python code is
  currently in contrib.
  """

    def __init__(self, name, table_ref=None):
        if False:
            return 10
        if table_ref is None:
            self.table_ref = gen_lookup_ops.mutable_hash_table_v2(key_dtype=dtypes.string, value_dtype=dtypes.float32, name=name)
        else:
            self.table_ref = table_ref
        self._name = name
        if not context.executing_eagerly():
            self._saveable = CheckpointedOp.CustomSaveable(self, name)
            ops_lib.add_to_collection(ops_lib.GraphKeys.SAVEABLE_OBJECTS, self._saveable)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    @property
    def saveable(self):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            return CheckpointedOp.CustomSaveable(self, self.name)
        else:
            return self._saveable

    def insert(self, keys, values):
        if False:
            print('Hello World!')
        return gen_lookup_ops.lookup_table_insert_v2(self.table_ref, keys, values)

    def lookup(self, keys, default):
        if False:
            print('Hello World!')
        return gen_lookup_ops.lookup_table_find_v2(self.table_ref, keys, default)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self._export()[0]

    def values(self):
        if False:
            while True:
                i = 10
        return self._export()[1]

    def _export(self):
        if False:
            return 10
        return gen_lookup_ops.lookup_table_export_v2(self.table_ref, dtypes.string, dtypes.float32)

    class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):
        """A custom saveable for CheckpointedOp."""

        def __init__(self, table, name):
            if False:
                while True:
                    i = 10
            tensors = table._export()
            specs = [saver_module.BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), saver_module.BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
            super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

        def restore(self, restore_tensors, shapes):
            if False:
                while True:
                    i = 10
            return gen_lookup_ops.lookup_table_import_v2(self.op.table_ref, restore_tensors[0], restore_tensors[1])