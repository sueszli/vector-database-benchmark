import unittest
from paddle.base import core

class TestPybindInference(unittest.TestCase):

    def test_get_op_attrs_default_value(self):
        if False:
            return 10
        core.get_op_attrs_default_value(b'fill_constant')
if __name__ == '__main__':
    unittest.main()