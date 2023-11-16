"""Tests for the pybind11 bindings of format_converter."""
from absl.testing import absltest
import numpy as np
from tensorflow.lite.tools.optimize.sparsity import format_converter_wrapper_pybind11 as format_converter

class FormatConverterTest(absltest.TestCase):

    def test_bcsr_fp32(self):
        if False:
            return 10
        'Same as FormatConverterTest::BlockTestD0S1 but via pybind11.'
        dense_matrix = [1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0]
        dense_shape = [4, 4]
        traversal_order = [0, 1, 2, 3]
        dim_types = [format_converter.TfLiteDimensionType.TF_LITE_DIM_DENSE, format_converter.TfLiteDimensionType.TF_LITE_DIM_SPARSE_CSR]
        block_size = [2, 2]
        block_map = [0, 1]
        converter = format_converter.FormatConverterFp32(dense_shape, traversal_order, dim_types, block_size, block_map)
        converter.DenseToSparse(np.asarray(dense_matrix, dtype=np.float32).data)
        dim_metadata = converter.GetDimMetadata()
        self.assertEqual([2], dim_metadata[0])
        self.assertEmpty(dim_metadata[1])
        self.assertEqual([0, 2, 3], dim_metadata[2])
        self.assertEqual([0, 1, 1], dim_metadata[3])
        self.assertEqual([2], dim_metadata[4])
        self.assertEmpty(dim_metadata[5])
        self.assertEqual([2], dim_metadata[6])
        self.assertEmpty(dim_metadata[7])
        expected_data = [1.0, 0.0, 0.0, 4.0, 2.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0]
        sparse_data = converter.GetData()
        self.assertTrue(np.allclose(expected_data, sparse_data))
        converter.SparseToDense(np.asarray(sparse_data, dtype=np.float32).data)
        self.assertTrue(np.allclose(dense_matrix, converter.GetData()))
if __name__ == '__main__':
    absltest.main()