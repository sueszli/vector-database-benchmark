import unittest
from paddle.base import framework

class ConstantTest(unittest.TestCase):

    def test_const_value(self):
        if False:
            while True:
                i = 10
        self.assertEqual(framework.GRAD_VAR_SUFFIX, '@GRAD')
        self.assertEqual(framework.TEMP_VAR_NAME, '@TEMP@')
        self.assertEqual(framework.GRAD_VAR_SUFFIX, '@GRAD')
        self.assertEqual(framework.ZERO_VAR_SUFFIX, '@ZERO')
if __name__ == '__main__':
    unittest.main()