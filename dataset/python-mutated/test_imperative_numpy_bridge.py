import unittest
import warnings
import numpy as np
from paddle import base

class TestImperativeNumpyBridge(unittest.TestCase):

    def test_tensor_from_numpy(self):
        if False:
            i = 10
            return i + 15
        data_np = np.array([[2, 3, 1]]).astype('float32')
        with base.dygraph.guard(base.CPUPlace()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                var = base.dygraph.to_variable(data_np, zero_copy=True)
                assert 'Currently, zero_copy is not supported, and it will be discarded.' in str(w[-1].message)
            var2 = base.dygraph.to_variable(data_np, zero_copy=False)
            np.testing.assert_array_equal(var2.numpy(), data_np)
            data_np[0][0] = -1
            self.assertEqual(data_np[0][0], -1)
            self.assertNotEqual(var2[0][0].numpy(), -1)
            self.assertFalse(np.array_equal(var2.numpy(), data_np))
if __name__ == '__main__':
    unittest.main()