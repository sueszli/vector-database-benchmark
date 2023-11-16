import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestUnsqueezeSPMDRule(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rule = core.get_phi_spmd_rule('unsqueeze')
        x_shape = [8, 16]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.attrs = OrderedDict()

    def test_unsqueeze_infer_forward(self):
        if False:
            while True:
                i = 10
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1])
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 0, 1])
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0])
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1, 0])
        self.x_dist_tensor_spec.shape = [1, 8, 16]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1])

    def test_unsqueeze_infer_backward(self):
        if False:
            print('Hello World!')
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [-1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh
        self.output_dist_tensor_spec = DistTensorSpec([8, 16], output_tensor_dist_attr)
        self.output_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.output_dist_tensor_spec.shape = [8, 16, 1]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.output_dist_tensor_spec.shape = [8, 1, 1, 16]
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1])
        self.output_dist_tensor_spec.shape = [1, 1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 0, 1])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 0, 1])
        self.output_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0])
        self.output_dist_tensor_spec.shape = [8, 16, 1]
        self.output_dist_tensor_spec.set_dims_mapping([1, 0, -1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.output_dist_tensor_spec.shape = [8, 1, 1, 16]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.output_dist_tensor_spec.shape = [1, 1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 1, 0])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1, 0])
        self.x_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.shape = [1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['axis'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1])
if __name__ == '__main__':
    unittest.main()