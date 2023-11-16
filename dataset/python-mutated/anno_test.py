"""Tests for anno module."""
import ast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test

class AnnoTest(test.TestCase):

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        node = ast.Name()
        self.assertEqual(anno.keys(node), set())
        self.assertFalse(anno.hasanno(node, 'foo'))
        with self.assertRaises(AttributeError):
            anno.getanno(node, 'foo')
        anno.setanno(node, 'foo', 3)
        self.assertEqual(anno.keys(node), {'foo'})
        self.assertTrue(anno.hasanno(node, 'foo'))
        self.assertEqual(anno.getanno(node, 'foo'), 3)
        self.assertEqual(anno.getanno(node, 'bar', default=7), 7)
        anno.delanno(node, 'foo')
        self.assertEqual(anno.keys(node), set())
        self.assertFalse(anno.hasanno(node, 'foo'))
        with self.assertRaises(AttributeError):
            anno.getanno(node, 'foo')
        self.assertIsNone(anno.getanno(node, 'foo', default=None))

    def test_copy(self):
        if False:
            print('Hello World!')
        node_1 = ast.Name()
        anno.setanno(node_1, 'foo', 3)
        node_2 = ast.Name()
        anno.copyanno(node_1, node_2, 'foo')
        anno.copyanno(node_1, node_2, 'bar')
        self.assertTrue(anno.hasanno(node_2, 'foo'))
        self.assertFalse(anno.hasanno(node_2, 'bar'))

    def test_duplicate(self):
        if False:
            print('Hello World!')
        node = ast.If(test=ast.Num(1), body=[ast.Expr(ast.Name('bar', ast.Load()))], orelse=[])
        anno.setanno(node, 'spam', 1)
        anno.setanno(node, 'ham', 1)
        anno.setanno(node.body[0], 'ham', 1)
        anno.dup(node, {'spam': 'eggs'})
        self.assertTrue(anno.hasanno(node, 'spam'))
        self.assertTrue(anno.hasanno(node, 'ham'))
        self.assertTrue(anno.hasanno(node, 'eggs'))
        self.assertFalse(anno.hasanno(node.body[0], 'eggs'))
if __name__ == '__main__':
    test.main()