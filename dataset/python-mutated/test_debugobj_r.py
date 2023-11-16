"""Test debugobj_r, coverage 56%."""
from idlelib import debugobj_r
import unittest

class WrappedObjectTreeItemTest(unittest.TestCase):

    def test_getattr(self):
        if False:
            return 10
        ti = debugobj_r.WrappedObjectTreeItem(list)
        self.assertEqual(ti.append, list.append)

class StubObjectTreeItemTest(unittest.TestCase):

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        ti = debugobj_r.StubObjectTreeItem('socket', 1111)
        self.assertEqual(ti.sockio, 'socket')
        self.assertEqual(ti.oid, 1111)
if __name__ == '__main__':
    unittest.main(verbosity=2)