"""Tests for prompt generation."""
import unittest
from IPython.core.prompts import LazyEvaluate

class PromptTests(unittest.TestCase):

    def test_lazy_eval_unicode(self):
        if False:
            print('Hello World!')
        u = u'ünicødé'
        lz = LazyEvaluate(lambda : u)
        self.assertEqual(str(lz), u)
        self.assertEqual(format(lz), u)

    def test_lazy_eval_nonascii_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        u = u'ünicødé'
        b = u.encode('utf8')
        lz = LazyEvaluate(lambda : b)
        self.assertEqual(str(lz), str(b))
        self.assertEqual(format(lz), str(b))

    def test_lazy_eval_float(self):
        if False:
            i = 10
            return i + 15
        f = 0.503
        lz = LazyEvaluate(lambda : f)
        self.assertEqual(str(lz), str(f))
        self.assertEqual(format(lz), str(f))
        self.assertEqual(format(lz, '.1'), '0.5')