from torch.testing._internal.common_utils import TestCase, run_tests

class TestImportTime(TestCase):

    def test_time_import_torch(self):
        if False:
            print('Hello World!')
        TestCase.runWithPytorchAPIUsageStderr('import torch')

    def test_time_cuda_device_count(self):
        if False:
            i = 10
            return i + 15
        TestCase.runWithPytorchAPIUsageStderr('import torch; torch.cuda.device_count()')
if __name__ == '__main__':
    run_tests()