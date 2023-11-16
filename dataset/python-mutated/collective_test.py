"""Tests for DTensor collectives."""
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
Layout = layout_lib.Layout
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]

class CollectiveTest(test_util.DTensorBaseTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(CollectiveTest, self).setUp()
        global_ids = test_util.create_device_ids_array((2, 1))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids, test_util.create_device_list((2, 1), device)) for device in ('CPU', 'GPU', 'TPU')}
        self.mesh = self.configTestMesh(mesh_dict)
        self.fully_replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
        self.first_dimension_sharded_layout_2d = Layout.batch_sharded(self.mesh, _MESH_DIM_X, 2)
        self.scalar_layout = Layout.replicated(self.mesh, rank=0)

    def testReduceOnBfloat16(self):
        if False:
            while True:
                i = 10
        self.skipForDeviceType(['GPU'], 'GPUs do not support bfloat16 collective reduce')
        self.skipForDeviceType(['TPU'], 'This test only needs to run on 2 cores.', unless_device_count_equals_to=2)
        a = constant_op.constant(np.array([[1, 2, 3, 4], [5.0, 6.0, 7.0, 8.0]]), dtype=dtypes.bfloat16)
        expected_result = math_ops.reduce_sum(a)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        dtensor_result = math_ops.reduce_sum(sharded_a)
        self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

    def testReduceOnInt32(self):
        if False:
            while True:
                i = 10
        a = constant_op.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.int32)
        expected_result = math_ops.reduce_sum(a)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        dtensor_result = math_ops.reduce_sum(sharded_a)
        self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

    def testReduceOnInt8(self):
        if False:
            return 10
        a = constant_op.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.int8)
        expected_result = math_ops.reduce_sum(a)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        dtensor_result = math_ops.reduce_sum(sharded_a)
        self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

    def testTwoReducesWithAssign(self):
        if False:
            while True:
                i = 10
        self.skipForPathways('TODO(b/260775095)')
        a = constant_op.constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.float32)
        b = constant_op.constant(np.array([[11, 12, 13, 4], [15, 16, 17, 18]]), dtype=dtypes.float32)
        expected_result = math_ops.reduce_sum(a) * math_ops.reduce_sum(b)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        sharded_b = api.relayout(b, self.first_dimension_sharded_layout_2d)
        sharded_v = d_variable.DVariable(sharded_b)

        @polymorphic_function.function
        def func(a, b):
            if False:
                return 10
            a1 = math_ops.reduce_sum(a, name='reducea')
            sharded_v.assign(a)
            b1 = math_ops.reduce_sum(b, name='reduceb')
            return a1 * b1
        with api.default_mesh(self.mesh):
            dtensor_result = func(sharded_a, sharded_b)
        self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

    @parameterized.named_parameters(('_all', math_ops.reduce_all), ('_any', math_ops.reduce_any))
    def testReduceOnBool(self, reduction):
        if False:
            print('Hello World!')
        self.skipForDeviceType(['GPU'], 'GPUs do not support int32 collective reduce')
        self.skipForDeviceType(['TPU'], 'This test only needs to run on 2 cores.', unless_device_count_equals_to=2)
        a = constant_op.constant(np.array([[True, False, False, True], [False, False, False, True]]), dtype=dtypes.bool)
        expected_result = reduction(a)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        dtensor_result = reduction(sharded_a)
        self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

    def testAllToAllOnBool(self):
        if False:
            print('Hello World!')
        self.skipForDeviceType(['GPU'], 'GPUs do not support int32 collective reduce')
        self.skipForDeviceType(['TPU'], 'This test only needs to run on 2 cores.', unless_device_count_equals_to=2)
        a = constant_op.constant(np.array([[True, False, False, True], [False, False, False, True]]), dtype=dtypes.bool)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)
        self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

    def testAllToAllOnInt32(self):
        if False:
            print('Hello World!')
        self.skipForDeviceType(['GPU'], 'GPUs do not support int32 StridedSliceXXX Ops')
        self.skipForDeviceType(['TPU'], 'This test only needs to run on 2 cores.', unless_device_count_equals_to=2)
        a = constant_op.constant(np.array([[1, 2], [3, 4]]), dtype=dtypes.int32)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)
        self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

    def testCollectiveOpsOnComplex64(self):
        if False:
            print('Hello World!')
        a = constant_op.constant(np.array([[1, 2 + 2j], [3 + 1j, 4 + 5j]]), dtype=dtypes.complex64)
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)
        self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

    def testCollectiveOpsOnComplex128(self):
        if False:
            print('Hello World!')
        self.skipForDeviceType(['TPU'], 'TPU does not support comolex128')
        expected_layout = Layout.inner_sharded(self.mesh, 'x', rank=2)
        initial_layout = Layout.batch_sharded(self.mesh, 'x', rank=2)
        a = constant_op.constant(np.array([[1, 2 + 2j], [3 + 1j, 4 + 5j]]), dtype=dtypes.complex128)
        sharded_a_initial = api.relayout(a, initial_layout)
        sharded_a = api.relayout(sharded_a_initial, expected_layout)
        api.check_layout(sharded_a, expected_layout)

    def testNoOpAllToAll(self):
        if False:
            while True:
                i = 10
        self.skipForDeviceType(['TPU'], 'This test only needs to run on 2 cores.', unless_device_count_equals_to=2)
        a = constant_op.constant(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), dtype=dtypes.float32)
        expected_result = a
        sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
        dtensor_result = api.relayout(sharded_a, self.fully_replicated_layout_2d)
        self.assertDTensorEqual(expected_result, self.fully_replicated_layout_2d, dtensor_result)

    def testDeviceIdTensorOnSplitHost(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_tpu_present():
            self.skipTest('This test only runs on TPUs.')
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((2, 4))
        local_ids = [0, 1, 4, 5, 2, 3, 6, 7]
        mesh = layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids, test_util.create_device_list((2, 4), 'TPU'), 'tpu_mesh')
        if not config.backend_is_pw():
            device = dtensor_device.DTensorDevice(meshes=[mesh])
            device.set_tpu_core_ids('tpu_mesh', local_ids)
        else:
            test_backend_util.config_test_mesh(mesh)
        layout_x = Layout.batch_sharded(mesh, _MESH_DIM_X, 2)
        layout_y = Layout.batch_sharded(mesh, _MESH_DIM_Y, 2)
        replica_ids = constant_op.constant(np.array([[0, 0, 0, 0], [1, 0, 0, 0]]), dtype=dtypes.int32)
        replica_ids = api.relayout(replica_ids, layout_x)
        ones = constant_op.constant(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), dtype=dtypes.int32)
        ones = api.relayout(ones, layout_y)

        @polymorphic_function.function
        def func(a, b):
            if False:
                return 10
            return math_ops.matmul(a, b)
        dtensor_result = func(replica_ids, ones)
        expected_result = [constant_op.constant([loc[_MESH_DIM_X]] * 4, dtype=dtypes.int32, shape=[1, 4]) for loc in mesh.local_device_locations()]
        self.assertEqual(api.fetch_layout(dtensor_result), layout_x)
        dtensor_result = [t.numpy() for t in api.unpack(dtensor_result)]
        self.assertAllEqual(expected_result, dtensor_result)

    def testDifferentShapesBetweenCalls(self):
        if False:
            for i in range(10):
                print('nop')
        self.skipForTfrt('b/269333905, TFRT cpu fails due to step_id not propagated.')
        self.skipForDeviceType(['TPU'], 'Known failure under TPU for legalization requires a static shape.')

        def produce_data(inputs, label):
            if False:
                i = 10
                return i + 15
            inputs = api.relayout(inputs, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1))
            label = api.relayout(label, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1))
            return (inputs, label)

        @polymorphic_function.function
        def train_fn(inputs, label):
            if False:
                while True:
                    i = 10
            (inputs, indices) = array_ops.unique(inputs)
            return math_ops.unsorted_segment_sum(label, indices, len(inputs))
        (input1, label1) = produce_data([6, 0, 6, 0], [1, 2, 3, 4])
        (input2, label2) = produce_data([2, 1, 2, 0], [1, 2, 3, 4])
        result1 = train_fn(input1, label1)
        result2 = train_fn(input2, label2)
        self.assertAllEqual(result1.numpy(), [4, 6])
        self.assertAllEqual(result2.numpy(), [4, 2, 4])

class CollectiveTestWithCustomMesh(test_util.DTensorBaseTest):

    def testGlobalAllReduceCombiner(self):
        if False:
            return 10
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((8,))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids, test_util.create_device_list((8,), device)) for device in ('CPU', 'GPU', 'TPU')}
        mesh = self.configTestMesh(mesh_dict)
        fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
        first_dimension_sharded_layout_2d = Layout.batch_sharded(mesh, _MESH_DIM_X, 2)

        @polymorphic_function.function
        def func(a, b):
            if False:
                while True:
                    i = 10
            a = math_ops.reduce_sum(a, axis=[0])
            b = math_ops.reduce_sum(b, axis=[0])
            return gen_math_ops.square(a) + gen_math_ops.square(b)
        row = constant_op.constant(np.array([[1.0, 2.0]]), dtype=dtypes.float32)
        a = array_ops.repeat(row, repeats=[8], axis=0)
        b = gen_array_ops.reverse_v2(a, axis=[1])
        expected_result = func(a, b)
        a = api.relayout(a, first_dimension_sharded_layout_2d)
        b = api.relayout(b, first_dimension_sharded_layout_2d)
        dtensor_result = func(a, b)
        self.assertDTensorEqual(expected_result, fully_replicated_layout_1d, dtensor_result)

    def testGlobalAllReduceCombinerDifferentReduce(self):
        if False:
            for i in range(10):
                print('nop')
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((8,))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids, test_util.create_device_list((8,), device)) for device in ('CPU', 'GPU', 'TPU')}
        mesh = self.configTestMesh(mesh_dict)
        fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
        first_dimension_sharded_layout_2d = Layout.batch_sharded(mesh, _MESH_DIM_X, 2)

        @polymorphic_function.function
        def func(a, b):
            if False:
                while True:
                    i = 10
            a = math_ops.reduce_sum(a, axis=[0])
            b = math_ops.reduce_mean(b, axis=[0])
            return gen_math_ops.square(a) + gen_math_ops.square(b)
        row = constant_op.constant(np.array([[1.0, 2.0]]), dtype=dtypes.float32)
        a = array_ops.repeat(row, repeats=[8], axis=0)
        b = gen_array_ops.reverse_v2(a, axis=[1])
        expected_result = func(a, b)
        a = api.relayout(a, first_dimension_sharded_layout_2d)
        b = api.relayout(b, first_dimension_sharded_layout_2d)
        dtensor_result = func(a, b)
        self.assertDTensorEqual(expected_result, fully_replicated_layout_1d, dtensor_result)

    def testSubgroupAllReduceCombiner(self):
        if False:
            print('Hello World!')
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((4, 2))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids, test_util.create_device_list((4, 2), device)) for device in ('CPU', 'GPU', 'TPU')}
        mesh = self.configTestMesh(mesh_dict)
        fully_sharded_layout_2d = Layout(_MESH_DIMS, mesh)

        @polymorphic_function.function
        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = math_ops.reduce_sum(a, axis=[0])
            b = math_ops.reduce_sum(b, axis=[0])
            return gen_math_ops.square(a) + gen_math_ops.square(b)
        row = constant_op.constant(np.array([[1.0, 2.0]]), dtype=dtypes.float32)
        a = array_ops.repeat(row, repeats=[8], axis=0)
        b = gen_array_ops.reverse_v2(a, axis=[1])
        expected_result = func(a, b)
        a = api.relayout(a, fully_sharded_layout_2d)
        b = api.relayout(b, fully_sharded_layout_2d)
        dtensor_result = func(a, b)
        self.assertDTensorEqual(expected_result, Layout([_MESH_DIM_Y], mesh), dtensor_result)

    def testMixedPrecisionAllReduce(self):
        if False:
            while True:
                i = 10
        has_enable_dtensor_mixed_precision_reduce = 'DTENSOR_ENABLE_MIXED_PRECISION_REDUCE' in os.environ
        has_dtensor_reduce_in_bfloat16_max_group_size = 'DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE' in os.environ
        if has_dtensor_reduce_in_bfloat16_max_group_size:
            old_dtensor_reduce_in_bfloat16_max_group_size = os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']
        os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE'] = ''
        os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = '4'
        self.skipForDeviceType(['GPU'], 'GPUs do not support bfloat16 reduce')
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((8,))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids, test_util.create_device_list((8,), device)) for device in ('CPU', 'GPU', 'TPU')}
        mesh = self.configTestMesh(mesh_dict)
        replicated_layout_1d = Layout.replicated(mesh, rank=1)
        first_dim_sharded_layout_1d = Layout.batch_sharded(mesh, _MESH_DIM_X, rank=2)

        @polymorphic_function.function
        def func(x):
            if False:
                while True:
                    i = 10
            return math_ops.reduce_sum(x, axis=0)
        inp = constant_op.constant(np.arange(48.0).reshape((8, 6)), dtype=dtypes.bfloat16)
        expected_result = np.sum(inp, axis=0)
        inp_dtensor = api.relayout(inp, first_dim_sharded_layout_1d)
        dtensor_result = func(inp_dtensor)
        self.assertDTensorEqual(expected_result, replicated_layout_1d, dtensor_result)
        if not has_enable_dtensor_mixed_precision_reduce:
            del os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE']
        if has_dtensor_reduce_in_bfloat16_max_group_size:
            os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = old_dtensor_reduce_in_bfloat16_max_group_size
        else:
            del os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']

    def testAllReduceCombinerWithIndirectDependency(self):
        if False:
            while True:
                i = 10
        self.skipForPathways('TODO(b/260775095)')
        self.skipForDeviceType(['TPU'], 'This test requires 8 TPU cores.', unless_device_count_equals_to=8)
        global_ids = test_util.create_device_ids_array((8,))
        local_ids = np.ravel(global_ids).tolist()
        mesh_dict = {device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids, test_util.create_device_list((8,), device)) for device in ('CPU', 'GPU', 'TPU')}
        mesh = self.configTestMesh(mesh_dict)
        first_dim_sharded_layout_1d = Layout.batch_sharded(mesh, _MESH_DIM_X, rank=1)
        init_value = constant_op.constant(np.ones(32), dtype=dtypes.float32)
        init_value = api.relayout(init_value, first_dim_sharded_layout_1d)

        @polymorphic_function.function
        def func(v):
            if False:
                print('Hello World!')
            a = math_ops.reduce_sum(v)
            v.assign_add(v + a)
            b = math_ops.reduce_sum(v)
            return b
        v = d_variable.DVariable(init_value)
        dtensor_result = func(v)
        expected_result = constant_op.constant(np.ones(32), dtype=dtypes.float32)
        expected_result += expected_result + math_ops.reduce_sum(expected_result)
        expected_result = math_ops.reduce_sum(expected_result)
        self.assertDTensorEqual(expected_result, Layout.replicated(mesh=mesh, rank=0), dtensor_result)
if __name__ == '__main__':
    test.main()