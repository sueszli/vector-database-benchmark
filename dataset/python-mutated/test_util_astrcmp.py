import unittest
from test.picardtestcase import PicardTestCase
from picard.util.astrcmp import astrcmp_py
try:
    from picard.util.astrcmp import astrcmp_c
except ImportError:
    astrcmp_c = None

class AstrcmpBase(object):
    func = None

    def test_astrcmp(self):
        if False:
            print('Hello World!')
        astrcmp = self.__class__.func
        self.assertAlmostEqual(0.0, astrcmp('', ''))
        self.assertAlmostEqual(0.0, astrcmp('a', ''))
        self.assertAlmostEqual(0.0, astrcmp('', 'a'))
        self.assertAlmostEqual(1.0, astrcmp('a', 'a'))
        self.assertAlmostEqual(0.0, astrcmp('a', 'b'))
        self.assertAlmostEqual(0.0, astrcmp('ab', 'ba'))
        self.assertAlmostEqual(0.7083333333333333, astrcmp('The Great Gig in the Sky', 'Great Gig In The sky'))

class AstrcmpCTest(AstrcmpBase, PicardTestCase):
    func = astrcmp_c

    @unittest.skipIf(astrcmp_c is None, 'The _astrcmp C extension module has not been compiled')
    def test_astrcmp(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_astrcmp()

class AstrcmpPyTest(AstrcmpBase, PicardTestCase):
    func = astrcmp_py