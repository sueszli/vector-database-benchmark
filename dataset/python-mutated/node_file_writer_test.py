import io
import os
import struct
import tempfile
import numpy as np
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework.test_util import IsMklEnabled
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
class NodeFileWriterTest(test.TestCase):
    """Tests for NodeFileWriter."""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.node_dir = tempfile.TemporaryDirectory(suffix='NodeFileWriterTest')
        os.environ['TF_NODE_FILE_WRITER_DIRECTORY'] = cls.node_dir.name
        with context.eager_mode():
            gen_math_ops.mat_mul(array_ops.ones((1, 1)), array_ops.ones((1, 1)))
        device = 'GPU' if config.list_physical_devices('GPU') else 'CPU'
        files_with_device = {file for file in os.listdir(cls.node_dir.name) if f'_{device}_0_' in file}
        assert len(files_with_device) == 1, f'Expected to create exactly one test_nodes file in directory {cls.node_dir.name} with string _{device}_0_ but found {len(files_with_device)}: {files_with_device}'
        (file,) = files_with_device
        assert file.startswith('node_defs_')
        cls.node_filename = os.path.join(cls.node_dir.name, file)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        super().tearDownClass()
        cls.node_dir.cleanup()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.node_file = open(self.node_filename, 'rb')
        self.node_file.seek(0, io.SEEK_END)

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        self.node_file.close()

    def _get_new_node_defs(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets new NodeDefs written by the NodeFileWriter.\n\n    Returns:\n      A list of new NodeDefs in the file written by NodeDefWriter since the last\n      time this method was called.\n    '
        node_def_bytes = self.node_file.read()
        node_defs = []
        cur_pos = 0
        while cur_pos < len(node_def_bytes):
            size_bytes = node_def_bytes[cur_pos:cur_pos + 8]
            (size,) = struct.unpack('<Q', size_bytes)
            cur_pos += 8
            node_def = node_def_pb2.NodeDef()
            node_def.ParseFromString(node_def_bytes[cur_pos:cur_pos + size])
            ignored_ops = []
            if context.run_eager_op_as_function_enabled():
                ignored_ops.extend(['_Arg', '_Retval', 'NoOp'])
                ignored_ops.extend(['_Recv', '_HostRecv'])
            if node_def.op not in ignored_ops:
                node_defs.append(node_def)
            cur_pos += size
        self.assertEqual(cur_pos, len(node_def_bytes))
        return node_defs

    def _get_input_shapes(self, node_def):
        if False:
            return 10
        input_shapes = []
        for shape_attr in node_def.attr['_input_shapes'].list.shape:
            shape = tuple((a.size for a in shape_attr.dim))
            input_shapes.append(shape)
        return input_shapes

    def _get_input_dtypes(self, node_def):
        if False:
            return 10
        input_dtypes = []
        for dtype_attr in node_def.attr['_input_dtypes'].list.type:
            input_dtypes.append(dtypes.as_dtype(dtype_attr))
        return input_dtypes

    def _get_input_tensor(self, node_def, input_index):
        if False:
            while True:
                i = 10
        tensor_proto = node_def.attr.get(f'_input_tensor_{input_index}')
        if tensor_proto is None:
            return None
        return tensor_util.MakeNdarray(tensor_proto.tensor)

    @test_util.disable_xla('b/201684914')
    def test_simple(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            x32 = constant_op.constant(np.ones((2, 3)).astype(np.float32))
            y32 = constant_op.constant(np.ones((3, 2)).astype(np.float32))
            x64 = constant_op.constant(np.ones((2, 3)).astype(np.float64))
            y64 = constant_op.constant(np.ones((3, 2)).astype(np.float64))
            gen_math_ops.mat_mul(x32, y32)
            gen_math_ops.mat_mul(x64, y64)
            node_defs = self._get_new_node_defs()
            self.assertLen(node_defs, 2)
            (node_def1, node_def2) = node_defs
            if not IsMklEnabled():
                self.assertEqual(node_def1.op, 'MatMul')
            else:
                self.assertIn(node_def1.op, ['MatMul', '_MklMatMul'])
            self.assertEqual(self._get_input_dtypes(node_def1), [dtypes.float32, dtypes.float32])
            self.assertEqual(self._get_input_shapes(node_def1), [(2, 3), (3, 2)])
            self.assertEqual(node_def2.op, 'MatMul')
            self.assertEqual(self._get_input_dtypes(node_def2), [dtypes.float64, dtypes.float64])
            self.assertEqual(self._get_input_shapes(node_def2), [(2, 3), (3, 2)])
            x32 = constant_op.constant(np.ones((4, 3)).astype(np.float32))
            gen_math_ops.mat_mul(x32, y32)
            node_defs = self._get_new_node_defs()
            self.assertLen(node_defs, 1)
            (node_def3,) = node_defs
            if not IsMklEnabled():
                self.assertEqual(node_def3.op, 'MatMul')
            else:
                self.assertIn(node_def3.op, ['MatMul', '_MklMatMul'])
            self.assertEqual(self._get_input_dtypes(node_def3), [dtypes.float32, dtypes.float32])
            self.assertEqual(self._get_input_shapes(node_def3), [(4, 3), (3, 2)])

    @test_util.disable_xla('b/201684914')
    def test_host_int32_inputs(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            x = constant_op.constant(np.ones((2, 2)).astype(np.float32))
            paddings = constant_op.constant([[1, 2], [3, 4]])
            constant_values = constant_op.constant(0.0)
            gen_array_ops.pad_v2(x, paddings, constant_values)
            node_defs = self._get_new_node_defs()
            self.assertLen(node_defs, 1)
            (node_def,) = node_defs
            self.assertEqual(node_def.op, 'PadV2')
            self.assertEqual(self._get_input_dtypes(node_def), [dtypes.float32, dtypes.int32, dtypes.float32])
            self.assertEqual(self._get_input_shapes(node_def), [(2, 2), (2, 2), ()])
            self.assertIsNone(self._get_input_tensor(node_def, 0))
            self.assertAllEqual(self._get_input_tensor(node_def, 1), np.array([[1, 2], [3, 4]]))
            self.assertIsNone(self._get_input_tensor(node_def, 2))

    @test_util.disable_xla('b/201684914')
    def test_skipped_ops(self):
        if False:
            return 10
        with context.eager_mode():
            x = constant_op.constant(np.ones((1, 1, 1, 1)).astype(np.float32))
            gen_math_ops.cast(x, dtypes.float64)
            self.assertEmpty(self._get_new_node_defs())
            gen_nn_ops.conv2d(x, x, [1, 1, 1, 1], 'SAME')
            y = constant_op.constant(np.zeros((1, 1, 1, 1)).astype(np.float32))
            gen_nn_ops.conv2d(x, y, [1, 1, 1, 1], 'SAME')
            self.assertLen(self._get_new_node_defs(), 1)
            x = constant_op.constant(np.ones((1, 1, 1, 1, 1, 1)).astype(np.float32))
            paddings = constant_op.constant(np.ones((6, 2)).astype(np.int32))
            constant_values = constant_op.constant(0.0)
            gen_array_ops.pad_v2(x, paddings, constant_values)
            self.assertEmpty(self._get_new_node_defs())
if __name__ == '__main__':
    test.main()