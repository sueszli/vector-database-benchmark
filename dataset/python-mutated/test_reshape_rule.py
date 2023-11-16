import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestReshapeSPMDRule(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.rule = core.get_phi_spmd_rule('reshape')
        x_shape = [6, 12, 48, 24]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.attrs = OrderedDict([('shape', [1, 72, 48, 4, 6])])

    def test_reshape_infer_forward(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1, -1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1])
        self.attrs['shape'] = [3, 24, 6, 8, 24]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1, 0])
        self.attrs['shape'] = [3, 24, 6, -1, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, -1, 1])
        self.attrs['shape'] = [1, 72, 0, 4, 6]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1])
        self.attrs['shape'] = [6, 12, 48, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.attrs['shape'] = [72, 3, 16, 24]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.attrs['shape'] = [72, 3, 16, 24]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1])
        self.x_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.attrs['shape'] = [6, 12, 48, 24]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.x_dist_tensor_spec.shape = [8, 1024, 3072]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1])
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1])
        self.attrs['shape'] = [3, 24, 6, -1, -1]
        with self.assertRaises(ValueError):
            self.rule.infer_forward(self.x_dist_tensor_spec, self.attrs['shape'])

    def test_reshape_infer_backward(self):
        if False:
            return 10
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh
        self.output_dist_tensor_spec = DistTensorSpec([1, 72, 48, 4, 6], output_tensor_dist_attr)
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1, -1, -1])
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1])
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, 0, -1])
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1, 0])
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, -1, 1])
        self.output_dist_tensor_spec.shape = [6, 12, 48, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1])
        self.output_dist_tensor_spec.shape = [72, 3, 16, 24]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.output_dist_tensor_spec.shape = [72, 3, 16, 24]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1])
        self.output_dist_tensor_spec.shape = [1, 72, 48, 4, 6]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1, 1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1, -1])
        self.output_dist_tensor_spec.shape = [3, 24, 6, 8, 24]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, 0])
        self.x_dist_tensor_spec.shape = [8, 1024, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1])
        self.x_dist_tensor_spec.shape = [-1, -1, 3072]
        self.output_dist_tensor_spec.shape = [0, 0, -1, 192]
        self.attrs['shape'] = [0, 0, -1, 192]
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, 1, -1])
        result_dist_attrs = self.rule.infer_backward(self.x_dist_tensor_spec, self.output_dist_tensor_spec, self.attrs['shape'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, 1, -1])
if __name__ == '__main__':
    unittest.main()