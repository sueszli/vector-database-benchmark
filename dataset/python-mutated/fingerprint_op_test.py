"""Tests for tensorflow.ops.fingerprint_op."""
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class FingerprintTest(test.TestCase):

    def test_default_values(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.arange(10)
        data = np.expand_dims(data, axis=0)
        fingerprint0 = self.evaluate(array_ops.fingerprint(data))
        fingerprint1 = self.evaluate(array_ops.fingerprint(data[:, 1:]))
        self.assertEqual(fingerprint0.ndim, 2)
        self.assertTupleEqual(fingerprint0.shape, fingerprint1.shape)
        self.assertTrue(np.any(fingerprint0 != fingerprint1))

    def test_empty(self):
        if False:
            print('Hello World!')
        f0 = self.evaluate(array_ops.fingerprint([]))
        self.assertEqual(f0.ndim, 2)
        self.assertEqual(f0.shape, (0, 8))
if __name__ == '__main__':
    test.main()