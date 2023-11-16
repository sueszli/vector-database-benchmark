"""Functional test for slot_creator."""
import numpy as np
from local_xla.xla import xla_data_pb2
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import slot_creator

def initialized_value(var):
    if False:
        print('Hello World!')
    return cond.cond(variable_v1.is_variable_initialized(var), var.read_value, lambda : var.initial_value)

class SlotCreatorTest(test.TestCase):

    def testCreateSlotFromVariable(self):
        if False:
            return 10
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            slot = slot_creator.create_slot(v, initialized_value(v), name='slot')
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('var/slot', slot.op.name)
            self.assertEqual([2], slot.get_shape().as_list())
            self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
            self.assertAllEqual([1.0, 2.5], self.evaluate(slot))

    def testCreateSlotFromTensor(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default(), self.cached_session():
            v = constant_op.constant([1.0, 2.5], name='const')
            slot = slot_creator.create_slot(v, v * 2, name='slot')
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('const/slot', slot.op.name)
            self.assertEqual([2], slot.get_shape().as_list())
            self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
            self.assertAllEqual([2.0, 5.0], self.evaluate(slot))

    def testCreateZerosSlotFromVariable(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('var/slot', slot.op.name)
            self.assertEqual([2], slot.get_shape().as_list())
            self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
            self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

    def testCreateZerosSlotFromDynamicShapedVariable(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default(), self.cached_session():
            dyn_shape = constant_op.constant([2], dtype=dtypes.int32)
            dyn_shape = array_ops.placeholder_with_default(dyn_shape, shape=[None])
            v = variable_scope.get_variable('var', initializer=random_ops.random_uniform(dyn_shape, dtype=dtypes.float64), validate_shape=False)
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('var/slot', slot.op.name)
            self.assertEqual([2], array_ops.shape(slot).eval())
            self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
            self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

    def testCreateZerosSlotFromTensor(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.cached_session():
            v = constant_op.constant([1.0, 2.5], name='const')
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot')
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('const/slot', slot.op.name)
            self.assertEqual([2], slot.get_shape().as_list())
            self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
            self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

    def testCreateZerosSlotFromDynamicShapedTensor(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default(), self.cached_session():
            v = random_ops.random_uniform([2], dtype=dtypes.float64)
            v = array_ops.placeholder_with_default(v, shape=[None], name='const')
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64)
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual('const/slot', slot.op.name)
            self.assertEqual([2], array_ops.shape(slot).eval())
            self.assertEqual(dtypes.float64, slot.dtype.base_dtype)
            self.assertAllEqual([0.0, 0.0], self.evaluate(slot))

    def testCreateSlotFromVariableRespectsScope(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default(), self.cached_session():
            with variable_scope.variable_scope('scope'):
                v = variables.Variable([1.0, 2.5], name='var')
                slot = slot_creator.create_slot(v, initialized_value(v), name='slot')
                self.assertEqual('scope/scope/var/slot', slot.op.name)

    def testCreateSlotFromFirstMDimensionVariable(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.test_session():
            s = variables.Variable([1.0, 2.5], name='var')
            p_v = variable_scope.get_variable('var', shape=[2, 2], partitioner=partitioned_variables.fixed_size_partitioner(2))
            for (i, v) in enumerate(p_v):
                slot = slot_creator.create_slot(v, initialized_value(s), name='slot')
                si = slot._save_slice_info
                self.evaluate(variables.global_variables_initializer())
                self.assertEqual('var/part_%d/slot' % i, slot.op.name)
                self.assertEqual([2], slot.get_shape().as_list())
                self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
                self.assertAllEqual([1.0, 2.5], slot)
                self.assertAllEqual([2], si.full_shape)
                self.assertAllEqual([i], si.var_offset)
                self.assertAllEqual([1], si.var_shape)

    def testCreateSlotFromScalarVariable(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.test_session():
            s = variables.Variable(1.0, name='var')
            p_v = variable_scope.get_variable('var', shape=[2, 2], partitioner=partitioned_variables.fixed_size_partitioner(2))
            for (i, v) in enumerate(p_v):
                slot = slot_creator.create_slot(v, initialized_value(s), name='slot')
                self.evaluate(variables.global_variables_initializer())
                self.assertEqual('var/part_%d/slot' % i, slot.op.name)
                self.assertEqual([], slot.get_shape().as_list())
                self.assertEqual(dtypes.float32, slot.dtype.base_dtype)
                self.assertAllEqual(1.0, slot)

    def testCreateSlotFromVariableCopyXlaSharding(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
            slot = slot_creator.create_slot(v, initialized_value(v), name='slot', copy_xla_sharding=True)
            self.assertEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(slot))

    def testCreateZerosSlotFromVariableCopyXlaSharding(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64, copy_xla_sharding=True)
            self.assertEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(slot))

    def testCreateSlotWithoutXlaSharding(self):
        if False:
            return 10
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
            with ops.control_dependencies(None):
                slot = slot_creator.create_slot(v, constant_op.constant(10, name='const'), name='slot', copy_xla_sharding=True)
            self.assertIsNone(xla_sharding.get_tensor_sharding(slot))
            self.assertNotEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(slot))

    def testCreateSlotWithCustomReplicatedXlaSharding(self):
        if False:
            return 10
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5], name='var')
            v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64, copy_xla_sharding=True)
                slot = xla_sharding.replicate(slot, use_sharding_op=False)
            self.assertNotEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(slot))
            slot_sharding = xla_sharding.get_tensor_sharding(slot)
            slot_proto = xla_data_pb2.OpSharding()
            slot_proto.ParseFromString(slot_sharding)
            self.assertEqual(slot_proto, xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED))

    def testCreateSlotWithCustomSplitXlaSharding(self):
        if False:
            return 10
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable([1.0, 2.5, 10.0, 15.1], name='var')
            v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
            with ops.control_dependencies(None):
                slot = slot_creator.create_zeros_slot(v, name='slot', dtype=dtypes.float64, copy_xla_sharding=True)
                slot = xla_sharding.split(slot, split_dimension=0, num_devices=4, use_sharding_op=False)
            self.assertNotEqual(xla_sharding.get_tensor_sharding(v), xla_sharding.get_tensor_sharding(slot))
            slot_sharding = xla_sharding.get_tensor_sharding(slot)
            slot_proto = xla_data_pb2.OpSharding()
            slot_proto.ParseFromString(slot_sharding)
            self.assertEqual(slot_proto, xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.OTHER, tile_assignment_dimensions=[4], tile_assignment_devices=range(4)))
if __name__ == '__main__':
    test.main()