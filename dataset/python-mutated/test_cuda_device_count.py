import unittest
import paddle

class TestDeviceCount(unittest.TestCase):

    def test_device_count(self):
        if False:
            print('Hello World!')
        s = paddle.device.cuda.device_count()
        self.assertIsNotNone(s)
if __name__ == '__main__':
    unittest.main()