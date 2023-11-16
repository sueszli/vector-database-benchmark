import torch
from torch.testing._internal.common_utils import TestCase, run_tests

class TestComparisonUtils(TestCase):

    def test_all_equal_no_assert(self):
        if False:
            i = 10
            return i + 15
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    def test_all_equal_no_assert_nones(self):
        if False:
            print('Hello World!')
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self):
        if False:
            while True:
                i = 10
        t = torch.tensor([0.5])
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    def test_assert_strides(self):
        if False:
            i = 10
            return i + 15
        t = torch.tensor([0.5])
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    def test_assert_sizes(self):
        if False:
            return 10
        t = torch.tensor([0.5])
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)
if __name__ == '__main__':
    run_tests()