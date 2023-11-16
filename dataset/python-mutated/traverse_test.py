"""Tests for Python module traversal."""
from tensorflow.python.platform import googletest
from tensorflow.tools.common import test_module1
from tensorflow.tools.common import test_module2
from tensorflow.tools.common import traverse

class TestVisitor(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.call_log = []

    def __call__(self, path, parent, children):
        if False:
            return 10
        self.call_log += [(path, parent, children)]

class TraverseTest(googletest.TestCase):

    def test_cycle(self):
        if False:
            return 10

        class Cyclist(object):
            pass
        Cyclist.cycle = Cyclist
        visitor = TestVisitor()
        traverse.traverse(Cyclist, visitor)

    def test_module(self):
        if False:
            for i in range(10):
                print('nop')
        visitor = TestVisitor()
        traverse.traverse(test_module1, visitor)
        called = [parent for (_, parent, _) in visitor.call_log]
        self.assertIn(test_module1.ModuleClass1, called)
        self.assertIn(test_module2.ModuleClass2, called)

    def test_class(self):
        if False:
            for i in range(10):
                print('nop')
        visitor = TestVisitor()
        traverse.traverse(TestVisitor, visitor)
        self.assertEqual(TestVisitor, visitor.call_log[0][1])
        self.assertIn('__init__', [name for (name, _) in visitor.call_log[0][2]])
        self.assertIn('__call__', [name for (name, _) in visitor.call_log[0][2]])

    def test_non_class(self):
        if False:
            i = 10
            return i + 15
        integer = 5
        visitor = TestVisitor()
        traverse.traverse(integer, visitor)
        self.assertEqual([], visitor.call_log)
if __name__ == '__main__':
    googletest.main()