import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestMatmulSPMDRule(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.rule = core.get_phi_spmd_rule('matmul')
        self.attrs = OrderedDict([('trans_x', False), ('trans_y', False)])

    def test_matmul_infer_forward(self):
        if False:
            i = 10
            return i + 15
        x_shape = [64, 32]
        y_shape = [32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {1})
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, True, False)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, False, True)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        infered_output_dist_attrs[0]._clean_partial_dims([0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, True, True)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True
        with self.assertRaises(NotImplementedError):
            result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])

    def test_matmul_infer_backward(self):
        if False:
            i = 10
            return i + 15
        x_shape = [64, 32]
        y_shape = [32, 48]
        out_shape = [64, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [-1, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [1, 0]
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(out_shape, out_tensor_dist_attr)
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[0]._is_partial(), False)
        self.assertEqual(infered_input_dist_attrs[1]._is_partial(), False)
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [1, 64, 32]
        self.y_dist_tensor_spec.shape = [512, 48, 32, 48]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1, 1, 0])
        self.out_dist_tensor_spec.set_dims_mapping([1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1, -1])
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.y_dist_tensor_spec.shape = [512, 1, 32, 48]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, -1, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, 1])
        self.out_dist_tensor_spec.shape = [512, 48, 64, 48]
        self.x_dist_tensor_spec.shape = [512, 48, 32, 64]
        self.y_dist_tensor_spec.shape = [512, 1, 48, 32]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, 0, 1])
        with self.assertRaises(RuntimeError):
            result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.y_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['trans_x'], self.attrs['trans_y'])
if __name__ == '__main__':
    unittest.main()