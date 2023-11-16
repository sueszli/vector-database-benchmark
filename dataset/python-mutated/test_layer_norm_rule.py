import unittest
from collections import OrderedDict
import numpy as np
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestLayerNormSPMDRule(unittest.TestCase):
    """
    Unit tests for layer_norm spmd rule.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.rule = core.get_phi_spmd_rule('layer_norm')
        x_shape = [64, 32, 1024]
        scale_shape = [1024]
        bias_shape = [1024]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.scale_spec = DistTensorSpec(self.x_spec)
        self.bias_spec = DistTensorSpec(self.x_spec)
        self.scale_spec.shape = scale_shape
        self.scale_spec.set_dims_mapping([-1])
        self.bias_spec.shape = bias_shape
        self.bias_spec.set_dims_mapping([-1])
        self.out_spec = DistTensorSpec(self.x_spec)
        self.mean_spec = DistTensorSpec(self.x_spec)
        self.var_spec = DistTensorSpec(self.x_spec)
        self.attrs = OrderedDict([('epsilon', 0.001), ('begin_norm_axis', 2)])

    def test_infer_forward(self):
        if False:
            return 10
        self.x_spec.set_dims_mapping([1, -1, -1])
        self.bias_spec.set_dims_mapping([-1])
        self.scale_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_forward(self.x_spec, self.scale_spec, self.bias_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [1, -1])
        self.x_spec.set_dims_mapping([1, 0, -1])
        self.scale_spec.set_dims_mapping([0])
        self.bias_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_forward(self.x_spec, self.scale_spec, self.bias_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [1, 0])
        self.attrs['begin_norm_axis'] = 1
        self.x_spec.set_dims_mapping([0, -1, -1])
        x_shape = self.x_spec.shape
        self.scale_spec.shape = [x_shape[1] * x_shape[2]]
        self.bias_spec.shape = [x_shape[1] * x_shape[2]]
        self.scale_spec.set_dims_mapping([-1])
        self.bias_spec.set_dims_mapping([1])
        self.mean_spec.shape = [x_shape[1]]
        self.var_spec.shape = [x_shape[1]]
        result_dist_attrs = self.rule.infer_forward(self.x_spec, self.scale_spec, self.bias_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0])

    def test_infer_backward(self):
        if False:
            return 10
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [1024]
        self.bias_spec.shape = [1024]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([1, -1, -1])
        self.mean_spec.set_dims_mapping([1, -1])
        self.var_spec.set_dims_mapping([1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [1, -1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([0, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, -1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([-1, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, 1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, 1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([-1, 1, -1])
        self.mean_spec.set_dims_mapping([-1, -1])
        self.var_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [-1, 1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([1, -1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, -1])
        with self.assertRaises(NotImplementedError):
            result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([-1, 1, -1])
        self.mean_spec.set_dims_mapping([0, -1])
        self.var_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, 1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([0, 1, -1])
        self.mean_spec.set_dims_mapping([-1, -1])
        self.var_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, 1])
        self.attrs['begin_norm_axis'] = 2
        self.scale_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.bias_spec.shape = [np.prod(self.x_spec.shape[self.attrs['begin_norm_axis']:])]
        self.mean_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.var_spec.shape = self.x_spec.shape[:self.attrs['begin_norm_axis']]
        self.out_spec.set_dims_mapping([0, -1, -1])
        self.mean_spec.set_dims_mapping([-1, 1])
        self.var_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_spec, self.scale_spec, self.bias_spec, self.out_spec, self.mean_spec, self.var_spec, self.attrs['epsilon'], self.attrs['begin_norm_axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 3)
        self.assertEqual(len(infered_output_dist_attrs), 3)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, 1])
if __name__ == '__main__':
    unittest.main()