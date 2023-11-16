import unittest

class Test(unittest.TestCase):

    def test_errors(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            raise ValueError
        with self.assertRaises(expected_exception=ValueError):
            raise ValueError
        with self.failUnlessRaises(ValueError):
            raise ValueError
        with self.assertRaisesRegex(ValueError, 'test'):
            raise ValueError('test')
        with self.assertRaisesRegex(ValueError, expected_regex='test'):
            raise ValueError('test')
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex='test'):
            raise ValueError('test')
        with self.assertRaisesRegex(expected_regex='test', expected_exception=ValueError):
            raise ValueError('test')
        with self.assertRaisesRegexp(ValueError, 'test'):
            raise ValueError('test')

    def test_unfixable_errors(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError, msg='msg'):
            raise ValueError
        with self.assertRaises(ValueError):
            raise ValueError
        with self.assertRaises(ValueError):
            raise ValueError