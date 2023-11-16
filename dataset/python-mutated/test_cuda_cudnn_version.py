import unittest
import paddle

class TestCPUVersion(unittest.TestCase):

    def test_cuda_cudnn_version_in_cpu_package(self):
        if False:
            i = 10
            return i + 15
        if not paddle.is_compiled_with_cuda():
            self.assertEqual(paddle.version.cuda(), 'False')
            self.assertEqual(paddle.version.cudnn(), 'False')
if __name__ == '__main__':
    unittest.main()