import unittest
import astor
import scipy
from paddle.jit import ignore_module
from paddle.jit.dy2static.convert_call_func import BUILTIN_LIKELY_MODULES

class TestIgnoreModule(unittest.TestCase):

    def test_ignore_module(self):
        if False:
            print('Hello World!')
        modules = [scipy, astor]
        ignore_module(modules)
        self.assertEqual([scipy, astor], BUILTIN_LIKELY_MODULES[-2:], 'Failed to add modules that ignore transcription')
if __name__ == '__main__':
    unittest.main()