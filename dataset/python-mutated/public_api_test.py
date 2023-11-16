"""Tests for tensorflow.tools.common.public_api."""
from tensorflow.python.platform import googletest
from tensorflow.tools.common import public_api

class PublicApiTest(googletest.TestCase):

    class TestVisitor(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.symbols = set()
            self.last_parent = None
            self.last_children = None

        def __call__(self, path, parent, children):
            if False:
                print('Hello World!')
            self.symbols.add(path)
            self.last_parent = parent
            self.last_children = list(children)

    def test_call_forward(self):
        if False:
            return 10
        visitor = self.TestVisitor()
        children = [('name1', 'thing1'), ('name2', 'thing2')]
        public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
        self.assertEqual(set(['test']), visitor.symbols)
        self.assertEqual('dummy', visitor.last_parent)
        self.assertEqual([('name1', 'thing1'), ('name2', 'thing2')], visitor.last_children)

    def test_private_child_removal(self):
        if False:
            print('Hello World!')
        visitor = self.TestVisitor()
        children = [('name1', 'thing1'), ('_name2', 'thing2')]
        public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
        self.assertEqual([('name1', 'thing1')], visitor.last_children)
        self.assertEqual([('name1', 'thing1')], children)

    def test_no_descent_child_removal(self):
        if False:
            print('Hello World!')
        visitor = self.TestVisitor()
        children = [('name1', 'thing1'), ('mock', 'thing2')]
        public_api.PublicAPIVisitor(visitor)('test', 'dummy', children)
        self.assertEqual([('name1', 'thing1'), ('mock', 'thing2')], visitor.last_children)
        self.assertEqual([('name1', 'thing1')], children)
if __name__ == '__main__':
    googletest.main()