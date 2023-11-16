import io
import numpy as np
import onnx
import pytorch_test_common
import torch
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion
from torch.onnx import _constants, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import jit_utils
from torch.testing._internal import common_utils

def expect_tensor(scalar_type, shape=None):
    if False:
        while True:
            i = 10

    def verify(actual_type):
        if False:
            print('Hello World!')
        np.testing.assert_equal(actual_type.scalarType(), scalar_type)
        if shape is not None:
            np.testing.assert_equal(actual_type.varyingSizes(), shape)
    return verify

def as_graphcontext(graph: torch.Graph) -> jit_utils.GraphContext:
    if False:
        return 10
    return jit_utils.GraphContext(graph=graph, block=graph.block(), opset=_constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET, original_node=None, params_dict={}, env={})

def g_op(graph: torch.Graph, op_name: str, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    return as_graphcontext(graph).op(op_name, *args, **kwargs)

class TestONNXShapeInference(pytorch_test_common.ExportTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        GLOBALS.export_onnx_opset_version = self.opset_version

    def run_test(self, g, n, type_assertion_funcs):
        if False:
            i = 10
            return i + 15
        if not isinstance(type_assertion_funcs, list):
            type_assertion_funcs = [type_assertion_funcs]
        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        for (out, type_assertion_func) in zip(n.outputs(), type_assertion_funcs):
            type_assertion_func(out.type())

    def create_empty_graph(self):
        if False:
            print('Hello World!')
        g = torch._C.Graph()
        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        return g

    def insert_tensor_constant(self, g, tensor):
        if False:
            i = 10
            return i + 15
        return g_op(g, 'Constant', value_t=tensor)

    def test_cast(self):
        if False:
            while True:
                i = 10
        g = self.create_empty_graph()
        input = g.addInput()
        cast_out = g_op(g, 'Cast', input, to_i=1)
        self.run_test(g, cast_out.node(), expect_tensor('Float'))

    def test_constant_of_shape(self):
        if False:
            return 10
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(1, 2, 3, 4))
        shape = g_op(g, 'Shape', constant)
        constant_of_shape = g_op(g, 'ConstantOfShape', shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', shape=(1, 2, 3, 4)))

    def test_constant_of_shape_static(self):
        if False:
            while True:
                i = 10
        rank = 4
        g = self.create_empty_graph()
        constants = [self.insert_tensor_constant(g, torch.tensor(i + 1)) for i in range(rank)]
        shape = g_op(g, 'prim::ListConstruct', *constants)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g_op(g, 'ConstantOfShape', shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', shape=(1, 2, 3, 4)))

    def test_constant_of_shape_dynamic(self):
        if False:
            print('Hello World!')
        rank = 4
        g = self.create_empty_graph()
        inputs = [g.addInput() for i in range(rank)]
        shape = g_op(g, 'prim::ListConstruct', *inputs)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g_op(g, 'ConstantOfShape', shape, value_t=torch.tensor([2.0]))
        self.run_test(g, constant_of_shape.node(), expect_tensor('Float', shape=(None, None, None, None)))

    def test_gather_dynamic_index(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16]))
        indices = g.addInput()
        indices.setType(indices.type().with_dtype(torch.int64).with_sizes([None]))
        output = g_op(g, 'Gather', input, indices, axis_i=1)
        self.run_test(g, output.node(), expect_tensor('Float', shape=[None, None, 16, 16]))

    def test_gather_scalar_index(self):
        if False:
            while True:
                i = 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16]))
        indices = self.insert_tensor_constant(g, torch.tensor(1))
        output = g_op(g, 'Gather', input, indices, axis_i=1)
        self.run_test(g, output.node(), expect_tensor('Float', shape=[None, 16, 16]))

    def test_reshape(self):
        if False:
            while True:
                i = 10
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 5))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([2, 0, -1]))
        shape = g_op(g, 'Reshape', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(2, 16, 25)))
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 4]))
        shape = g_op(g, 'Reshape', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(10, 16, 4)))
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 0]))
        shape = g_op(g, 'Reshape', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(8, 16, 5)))

    def test_reshape_symbolic(self):
        if False:
            i = 10
            return i + 15
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None, 2, 8]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 0, -1]))
        output = g_op(g, 'Reshape', input, constant)
        self.run_test(g, output.node(), expect_tensor(None, shape=(None, None, 16)))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        if False:
            print('Hello World!')
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([3, 4, 0]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 4, 3]))
        output = g_op(g, 'Reshape', input, constant, allowzero_i=1)
        self.run_test(g, output.node(), expect_tensor(None, shape=(0, 4, 3)))

    def test_slice(self):
        if False:
            return 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None]))
        start_input = g.addInput()
        start_input.setType(start_input.type().with_sizes([None]))
        end = self.insert_tensor_constant(g, torch.tensor([3]))
        axis = self.insert_tensor_constant(g, torch.tensor([0]))
        step = self.insert_tensor_constant(g, torch.tensor([1]))
        slice = g_op(g, 'Slice', input, start_input, end, axis, step)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(None, None)))

    def test_slice_with_dynamic_start_index(self):
        if False:
            return 10
        g = self.create_empty_graph()
        input = self.insert_tensor_constant(g, torch.ones(2, 3, 4, 5))
        start_input = g.addInput()
        start_input.setType(start_input.type().with_sizes([2]))
        end = self.insert_tensor_constant(g, torch.tensor([3, 4]))
        axis = self.insert_tensor_constant(g, torch.tensor([1, -1]))
        slice = g_op(g, 'Slice', input, start_input, end, axis)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(2, None, 4, None)))

    def test_broadcast_matmul(self):
        if False:
            i = 10
            return i + 15
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, 'MatMul', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(3, 5, 1, 1)))
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, 'MatMul', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(3, 1, 1)))
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, 'MatMul', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=(5, 1)))
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, 'MatMul', constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor('Float', shape=()))

    def test_expand(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.create_empty_graph()
        input = g.addInput()
        constant = self.insert_tensor_constant(g, torch.ones(2, 4))
        input.setType(constant.type().with_sizes([None, None]))
        shape = g_op(g, 'Shape', input)
        expand = g_op(g, 'Expand', constant, shape)
        self.run_test(g, expand.node(), expect_tensor('Float', shape=(None, None)))

    def test_pad(self):
        if False:
            while True:
                i = 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g_op(g, 'prim::Constant').setType(torch.NoneType.get())
        pad = g_op(g, 'Pad', input, constant, none, mode_s='constant')
        self.run_test(g, pad.node(), expect_tensor('Float', shape=(5, 322, 102)))

    def test_pad_with_dynamic_input_shape(self):
        if False:
            return 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, None, None]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g_op(g, 'prim::Constant').setType(torch.NoneType.get())
        pad = g_op(g, 'Pad', input, constant, none, mode_s='constant')
        self.run_test(g, pad.node(), expect_tensor('Float', shape=(5, None, None)))

    def test_pad_with_dynamic_pad_size(self):
        if False:
            print('Hello World!')
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        pad_size = g.addInput()
        pad_size.setType(pad_size.type().with_dtype(torch.long).with_sizes([6]))
        none = g_op(g, 'prim::Constant').setType(torch.NoneType.get())
        pad = g_op(g, 'Pad', input, pad_size, none, mode_s='constant')
        self.run_test(g, pad.node(), expect_tensor('Float', shape=(None, None, None)))

    def test_resize(self):
        if False:
            i = 10
            return i + 15
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g_op(g, 'prim::Constant').setType(torch.NoneType.get())
        scales = self.insert_tensor_constant(g, torch.tensor([1, 1, 2, 2], dtype=torch.float))
        resize = g_op(g, 'Resize', input, none, scales, coordinate_transformation_mode_s='align_corners', cubic_coeff_a_f=-0.75, mode_s='linear', nearest_mode_s='floor')
        self.run_test(g, resize.node(), expect_tensor('Float', shape=(4, 32, 128, 128)))

    def test_resize_after_concat(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g_op(g, 'prim::Constant').setType(torch.NoneType.get())
        scale_1 = self.insert_tensor_constant(g, torch.tensor([1, 1], dtype=torch.float))
        scale_2 = self.insert_tensor_constant(g, torch.tensor([2, 2], dtype=torch.float))
        scales = g_op(g, 'Concat', scale_1, scale_2, axis_i=0)
        resize = g_op(g, 'Resize', input, none, scales, coordinate_transformation_mode_s='align_corners', cubic_coeff_a_f=-0.75, mode_s='linear', nearest_mode_s='floor')
        self.run_test(g, resize.node(), expect_tensor('Float', shape=(4, 32, 128, 128)))

    def test_reduce_prod_with_axes(self):
        if False:
            while True:
                i = 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g_op(g, 'ReduceProd', input, axes_i=[0])
        self.run_test(g, reduce_prod.node(), expect_tensor('Long', shape=(1,)))

    def test_reduce_prod_without_axes(self):
        if False:
            i = 10
            return i + 15
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g_op(g, 'ReduceProd', input)
        self.run_test(g, reduce_prod.node(), expect_tensor('Long', shape=(1,)))

    def test_proceeding_nodes_use_prim_pack_padded_output_dtype_correctly(self):
        if False:
            return 10
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 16]))
        length = g.addInput()
        length.setType(length.type().with_dtype(torch.long).with_sizes([4]))
        (padded, batch_size) = g_op(g, 'prim::PackPadded', input, length, outputs=2)
        padded.setType(padded.type().with_dtype(torch.float).with_sizes([None, None]))
        batch_size.setType(batch_size.type().with_dtype(torch.long).with_sizes([None]))
        gather_idx = self.insert_tensor_constant(g, torch.tensor([0], dtype=torch.long))
        gather = g_op(g, 'Gather', batch_size, gather_idx, axis_i=0)
        self.run_test(g, gather.node(), expect_tensor('Long', shape=(None,)))

    def test_squeeze_after_dynamic_if(self):
        if False:
            return 10
        from torch.onnx.symbolic_opset11 import squeeze as squeeze11
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([1, None, 5]))
        cond = g.addInput()
        cond.setType(input.type().with_dtype(torch.int32).with_sizes([1]))
        (if_op, (if_context, else_context), new_node) = jit_utils.add_op_with_blocks(as_graphcontext(g), 'If', cond, n_blocks=2)
        block1_output = if_context.op('Add', input, input)
        block2_output = else_context.op('Identity', input)
        utils._add_output_to_block(if_context.block, block1_output)
        utils._add_output_to_block(else_context.block, block2_output)
        if_output = torch._C._jit_pass_fixup_onnx_controlflow_node(new_node, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET)[0]
        torch._C._jit_pass_onnx_node_shape_type_inference(new_node, {}, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET)
        squeezed = squeeze11(as_graphcontext(g), if_output, dim=0)
        assert squeezed.node().kind() == 'onnx::Squeeze'
        self.run_test(g, squeezed.node(), expect_tensor('Float', shape=(None, 5)))

class TestONNXCustomOpShapeInference(pytorch_test_common.ExportTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET

    def test_setType_maintains_output_shape_for_single_custom_op(self):
        if False:
            while True:
                i = 10
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, '::linalg_inv', 9)

        class CustomInverse(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.inverse(x) + x

        def linalg_inv_settype(g, self):
            if False:
                return 10
            return g.op('com.microsoft::Inverse', self).setType(self.type())
        torch.onnx.register_custom_op_symbolic('::linalg_inv', linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version, custom_opsets={'com.microsoft': 1})
        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        assert model_value_info
        dims = model_value_info[0].type.tensor_type.shape.dim
        for i in range(len(dims)):
            self.assertTrue(dims[i].HasField('dim_value'))
        for (dim, rank) in zip(dims, x.size()):
            self.assertEqual(dim.dim_value, rank)

    def test_no_setType_for_single_custom_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, '::linalg_inv', 9)

        class CustomInverse(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.inverse(x) + x

        def linalg_inv_no_settype(g, self):
            if False:
                return 10
            return g.op('com.microsoft::Inverse', self)
        torch.onnx.register_custom_op_symbolic('::linalg_inv', linalg_inv_no_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version, custom_opsets={'com.microsoft': 1})
        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        assert model_value_info
        dims = model_value_info[0].type.tensor_type.shape.dim
        for i in range(len(dims)):
            self.assertTrue(dims[i].HasField('dim_param'))

    def test_setType_maintains_output_shape_for_single_custom_op_with_dynamic_axes(self):
        if False:
            return 10
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, '::linalg_inv', 9)

        class CustomInverse(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.inverse(x) + x

        def linalg_inv_settype(g, self):
            if False:
                print('Hello World!')
            return g.op('com.microsoft::Inverse', self).setType(self.type().with_dtype(torch.float).with_sizes([None, 3, 3]))
        torch.onnx.register_custom_op_symbolic('::linalg_inv', linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version, custom_opsets={'com.microsoft': 1}, input_names=['x'], dynamic_axes={'x': {0: 'batch'}})
        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        assert model_value_info
        dims = model_value_info[0].type.tensor_type.shape.dim
        self.assertTrue(dims[0].HasField('dim_param'))
        for i in range(1, len(dims)):
            self.assertTrue(dims[i].HasField('dim_value'))
            self.assertEqual(dims[i].dim_value, x.size()[i])

    def test_setType_maintains_output_shape_for_single_custom_op_with_onnx_ops(self):
        if False:
            i = 10
            return i + 15
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, '::linalg_inv', 9)

        class CustomInverse(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    for i in range(10):
                        print('nop')
                x = torch.inverse(x)
                return x + y + z

        def linalg_inv_settype(g, self):
            if False:
                i = 10
                return i + 15
            return g.op('com.microsoft::Inverse', self).setType(self.type().with_dtype(torch.float).with_sizes([2, 3, 10, 10]))
        torch.onnx.register_custom_op_symbolic('::linalg_inv', linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 10, 10)
        y = torch.randn(2, 3, 10, 10)
        z = torch.randn(2, 3, 10, 10)
        f = io.BytesIO()
        torch.onnx.export(model, (x, y, z), f, opset_version=self.opset_version, custom_opsets={'com.microsoft': 1})
        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        output_name = ''
        for node in model_proto.graph.node:
            if node.op_type == 'Inverse':
                output_name = node.output[0]
                break
        assert output_name
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        assert model_value_info
        for value_info in model_value_info:
            assert value_info.name
            if value_info.name == output_name:
                dims = value_info.type.tensor_type.shape.dim
                for i in range(len(dims)):
                    self.assertTrue(dims[i].HasField('dim_value'))
                for (dim, rank) in zip(dims, x.size()):
                    self.assertEqual(dim.dim_value, rank)
if __name__ == '__main__':
    common_utils.run_tests()