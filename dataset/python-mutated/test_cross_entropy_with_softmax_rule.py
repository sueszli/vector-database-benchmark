import unittest
from paddle.distributed.auto_parallel.static.completion import get_spmd_rule
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto

class TestCrossEntropyWithSoftmaxSPMDRule(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rule1 = get_spmd_rule('cross_entropy_with_softmax')
        x_shape = [8, 1024, 50304]
        label_shape = [8, 1024, 1]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        label_tensor_dist_attr = TensorDistAttr()
        label_tensor_dist_attr.process_mesh = process_mesh
        self.lable_dist_tensor_spec = DistTensorSpec(label_shape, label_tensor_dist_attr)
        self.loss_spec = DistTensorSpec(self.lable_dist_tensor_spec)
        self.softmax_out_spec = DistTensorSpec(self.x_dist_tensor_spec)
        self.attrs = {'ignore_index': -1, 'axis': -1, 'numeric_stable_mode': True, 'use_softmax': True, 'soft_label': False}

    def test_cross_entropy_with_softmax_infer_forward(self):
        if False:
            while True:
                i = 10
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, 0, -1])
        result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        self.assertEqual(len(result_dist_attrs), 2)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 2)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.attrs['soft_label'] = True
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        self.attrs['soft_label'] = False
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.attrs['axis'] = -1
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs)
        self.attrs['axis'] = -1

    def test_cross_entropy_with_softmax_infer_backward(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([1, 0, -1])
        self.loss_spec.set_dims_mapping([1, 0, -1])
        result_dist_attrs = self.rule1.infer_backward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], [self.softmax_out_spec, self.loss_spec], self.attrs)
        self.assertEqual(len(result_dist_attrs), 2)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 2)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, 0, -1])
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([-1, -1, 0])
        self.loss_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], [self.softmax_out_spec, self.loss_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [-1, -1, -1])
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([-1, -1, 0])
        self.loss_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], [self.softmax_out_spec, self.loss_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, -1])
        self.attrs['axis'] = 1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = True
        self.softmax_out_spec.set_dims_mapping([1, -1, 0])
        self.loss_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward([self.x_dist_tensor_spec, self.lable_dist_tensor_spec], [self.softmax_out_spec, self.loss_spec], self.attrs)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, 0])
if __name__ == '__main__':
    unittest.main()