import unittest
from paddle.base import core

class TestTF32Switch(unittest.TestCase):

    def test_on_off(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            self.assertTrue(core.get_cudnn_switch())
            core.set_cudnn_switch(0)
            self.assertFalse(core.get_cudnn_switch())
            core.set_cudnn_switch(1)
            self.assertTrue(core.get_cudnn_switch())
            core.set_cudnn_switch(1)
        else:
            pass
if __name__ == '__main__':
    unittest.main()