import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestReductionSPMDRule(unittest.TestCase):
    """
    Unit tests for reduction spmd rule.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.rule = core.get_phi_spmd_rule('max')
        x_shape = [64, 32]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)
        self.attrs = OrderedDict([('axis', [0]), ('keep_dim', False)])

    def test_single_mesh_dim(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})

    def test_multi_mesh_dim(self):
        if False:
            for i in range(10):
                print('nop')
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0, 1})
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {1})
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {1})

    def test_backward_single_mesh_dim(self):
        if False:
            i = 10
            return i + 15
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0]
        self.out_dist_tensor_spec.shape = [32]
        self.out_dist_tensor_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0]
        self.out_dist_tensor_spec.shape = [1, 32]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1]
        self.out_dist_tensor_spec.shape = [64]
        self.out_dist_tensor_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1]
        self.out_dist_tensor_spec.shape = [64, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0, 1]
        self.out_dist_tensor_spec.shape = []
        self.out_dist_tensor_spec.set_dims_mapping([])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [])
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0, 1]
        self.out_dist_tensor_spec.shape = [1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])

    def test_backward_multi_mesh_dim(self):
        if False:
            while True:
                i = 10
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.out_dist_tensor_spec.set_process_mesh(process_mesh)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1])
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96, 1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])

    def test_backward_multi_mesh_dim_parital(self):
        if False:
            i = 10
            return i + 15
        out_shape = [96, 1, 1]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [0, -1, -1]
        out_tensor_dist_attr.process_mesh = process_mesh
        out_tensor_dist_attr._set_partial_dims([1])
        self.out_dist_tensor_spec = DistTensorSpec(out_shape, out_tensor_dist_attr)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[0]._is_partial(), False)
if __name__ == '__main__':
    unittest.main()