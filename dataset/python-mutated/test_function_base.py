import pytest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO, TestCase
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal

class TestAppend(TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        result = np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
        assert_equal(result, np.arange(1, 10, dtype=int))
        result = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        assert_equal(result, np.arange(1, 10, dtype=int).reshape((3, 3)))
        with pytest.raises((RuntimeError, ValueError)):
            np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
if __name__ == '__main__':
    run_tests()