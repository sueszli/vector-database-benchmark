import unittest
from collections import OrderedDict
from paddle.distributed.auto_parallel.static.dist_attribute import DistTensorSpec, TensorDistAttr
from paddle.distributed.fleet import auto
from paddle.framework import core

class TestEmbeddingSPMDRule(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rule1 = core.get_phi_spmd_rule('lookup_table_v2')

    def test_embedding_infer_forward(self):
        if False:
            for i in range(10):
                print('nop')
        x_shape = [4, 1024]
        table_shape = [512, 768]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(table_shape, table_tensor_dist_attr)
        self.attrs = OrderedDict([('padding_idx', -1), ('sparse', False)])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule1.infer_forward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule1.infer_forward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule1.infer_forward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = 128
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward([self.x_dist_tensor_spec, self.table_dist_tensor_spec], self.attrs)
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = -1
        self.attrs['sparse'] = True
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])

    def test_embedding_infer_backward(self):
        if False:
            print('Hello World!')
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_shape = [4, 1024]
        table_shape = [512, 768]
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(table_shape, table_tensor_dist_attr)
        out_shape = [4, 1024, 768]
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(out_shape, out_tensor_dist_attr)
        self.attrs = OrderedDict([('padding_idx', -1), ('sparse', False)])
        self.out_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.out_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule1.infer_backward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.out_dist_tensor_spec.set_dims_mapping([1, 0, -1])
        result_dist_attrs = self.rule1.infer_backward(self.x_dist_tensor_spec, self.table_dist_tensor_spec, self.out_dist_tensor_spec, self.attrs['padding_idx'], self.attrs['sparse'])
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])
if __name__ == '__main__':
    unittest.main()