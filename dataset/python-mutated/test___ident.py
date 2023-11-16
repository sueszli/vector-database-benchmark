from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import gevent.testing as greentest
from gevent._ident import IdentRegistry
from gevent._compat import PYPY

class Target(object):
    pass

class TestIdent(greentest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.reg = IdentRegistry()

    def tearDown(self):
        if False:
            return 10
        self.reg = None

    def test_basic(self):
        if False:
            while True:
                i = 10
        target = Target()
        self.assertEqual(0, self.reg.get_ident(target))
        self.assertEqual(1, len(self.reg))
        self.assertEqual(0, self.reg.get_ident(target))
        self.assertEqual(1, len(self.reg))
        target2 = Target()
        self.assertEqual(1, self.reg.get_ident(target2))
        self.assertEqual(2, len(self.reg))
        self.assertEqual(1, self.reg.get_ident(target2))
        self.assertEqual(2, len(self.reg))
        self.assertEqual(0, self.reg.get_ident(target))
        del target
        if PYPY:
            for _ in range(3):
                gc.collect()
        self.assertEqual(1, len(self.reg))
        target3 = Target()
        self.assertEqual(1, self.reg.get_ident(target2))
        self.assertEqual(0, self.reg.get_ident(target3))
        self.assertEqual(2, len(self.reg))

    @greentest.skipOnPyPy('This would need to GC very frequently')
    def test_circle(self):
        if False:
            while True:
                i = 10
        keep_count = 3
        keepalive = [None] * keep_count
        for i in range(1000):
            target = Target()
            keepalive[i % keep_count] = target
            self.assertLessEqual(self.reg.get_ident(target), keep_count)

@greentest.skipOnPurePython('Needs C extension')
class TestCExt(greentest.TestCase):

    def test_c_extension(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(IdentRegistry.__module__, 'gevent._gevent_c_ident')
if __name__ == '__main__':
    greentest.main()