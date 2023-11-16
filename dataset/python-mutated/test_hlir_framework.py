import unittest
import numpy as np
from cinn import Target
from cinn.framework import Tensor

class TensorTest(unittest.TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        target = Target()
        target.arch = Target.Arch.X86
        target.bits = Target.Bit.k64
        target.os = Target.OS.Linux
        tensor = Tensor()
        data = np.random.random([10, 5])
        tensor.from_numpy(data, target)
        np.testing.assert_allclose(tensor.numpy(), data)
if __name__ == '__main__':
    unittest.main()