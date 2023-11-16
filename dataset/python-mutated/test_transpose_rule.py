import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestTransposeSPMDRule(unittest.TestCase):
    """
    Unit tests for reduction spmd rule.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule = core.get_phi_spmd_rule('transpose')
        x_shape = [64, 36]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)
        self.attrs = OrderedDict([('perm', [0, 1, 2, 3])])

    def test_single_mesh_dim(self):
        if False:
            print('Hello World!')
        self.attrs['perm'] = [1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.attrs['perm'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1])

    def test_multi_mesh_dim(self):
        if False:
            i = 10
            return i + 15
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0])
        self.attrs['perm'] = [0, 2, 3, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.attrs['perm'] = [-1, 0, -2, 1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1])

    def test_backward_single_mesh_dim(self):
        if False:
            return 10
        self.attrs['perm'] = [1, 0]
        self.out_dist_tensor_spec.shape = [36, 64]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.attrs['perm'] = [0, 1]
        self.out_dist_tensor_spec.shape = [64, 36]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.shape = [64, 36, 24, 48]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1])

    def test_backward_multi_mesh_dim(self):
        if False:
            print('Hello World!')
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.out_dist_tensor_spec.set_process_mesh(process_mesh)
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.shape = [64, 36, 24, 48]
        self.out_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0])
        self.attrs['perm'] = [0, 2, 3, 1]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.x_dist_tensor_spec.shape = [64, 48, 36, 24]
        self.attrs['perm'] = [-1, 0, -2, 1]
        self.out_dist_tensor_spec.shape = [24, 64, 36, 48]
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['perm'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1])
if __name__ == '__main__':
    unittest.main()