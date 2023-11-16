"""Test debugobj, coverage 40%."""
from idlelib import debugobj
import unittest

class ObjectTreeItemTest(unittest.TestCase):

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        ti = debugobj.ObjectTreeItem('label', 22)
        self.assertEqual(ti.labeltext, 'label')
        self.assertEqual(ti.object, 22)
        self.assertEqual(ti.setfunction, None)

class ClassTreeItemTest(unittest.TestCase):

    def test_isexpandable(self):
        if False:
            i = 10
            return i + 15
        ti = debugobj.ClassTreeItem('label', 0)
        self.assertTrue(ti.IsExpandable())

class AtomicObjectTreeItemTest(unittest.TestCase):

    def test_isexpandable(self):
        if False:
            for i in range(10):
                print('nop')
        ti = debugobj.AtomicObjectTreeItem('label', 0)
        self.assertFalse(ti.IsExpandable())

class SequenceTreeItemTest(unittest.TestCase):

    def test_isexpandable(self):
        if False:
            print('Hello World!')
        ti = debugobj.SequenceTreeItem('label', ())
        self.assertFalse(ti.IsExpandable())
        ti = debugobj.SequenceTreeItem('label', (1,))
        self.assertTrue(ti.IsExpandable())

    def test_keys(self):
        if False:
            return 10
        ti = debugobj.SequenceTreeItem('label', 'abc')
        self.assertEqual(list(ti.keys()), [0, 1, 2])

class DictTreeItemTest(unittest.TestCase):

    def test_isexpandable(self):
        if False:
            return 10
        ti = debugobj.DictTreeItem('label', {})
        self.assertFalse(ti.IsExpandable())
        ti = debugobj.DictTreeItem('label', {1: 1})
        self.assertTrue(ti.IsExpandable())

    def test_keys(self):
        if False:
            print('Hello World!')
        ti = debugobj.DictTreeItem('label', {1: 1, 0: 0, 2: 2})
        self.assertEqual(ti.keys(), [0, 1, 2])
if __name__ == '__main__':
    unittest.main(verbosity=2)