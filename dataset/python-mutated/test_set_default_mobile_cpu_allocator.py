import torch
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSetDefaultMobileCPUAllocator(TestCase):

    def test_no_exception(self):
        if False:
            while True:
                i = 10
        torch._C._set_default_mobile_cpu_allocator()
        torch._C._unset_default_mobile_cpu_allocator()

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            torch._C._unset_default_mobile_cpu_allocator()
        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._set_default_mobile_cpu_allocator()
        torch._C._unset_default_mobile_cpu_allocator()
        with self.assertRaises(Exception):
            torch._C._set_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()
            torch._C._unset_default_mobile_cpu_allocator()
if __name__ == '__main__':
    run_tests()