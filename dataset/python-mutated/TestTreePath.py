import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes

class TestTreePath(TransformTest):
    _tree = None

    def _build_tree(self):
        if False:
            print('Hello World!')
        if self._tree is None:
            self._tree = self.run_pipeline([], u'\n            def decorator(fun):  # DefNode\n                return fun       # ReturnStatNode, NameNode\n            @decorator           # NameNode\n            def decorated():     # DefNode\n                pass\n            ')
        return self._tree

    def test_node_path(self):
        if False:
            for i in range(10):
                print('nop')
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, '//DefNode')))
        self.assertEqual(2, len(find_all(t, '//NameNode')))
        self.assertEqual(1, len(find_all(t, '//ReturnStatNode')))
        self.assertEqual(1, len(find_all(t, '//DefNode//ReturnStatNode')))

    def test_node_path_star(self):
        if False:
            i = 10
            return i + 15
        t = self._build_tree()
        self.assertEqual(10, len(find_all(t, '//*')))
        self.assertEqual(8, len(find_all(t, '//DefNode//*')))
        self.assertEqual(0, len(find_all(t, '//NameNode//*')))

    def test_node_path_attribute(self):
        if False:
            while True:
                i = 10
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, '//NameNode/@name')))
        self.assertEqual(['fun', 'decorator'], find_all(t, '//NameNode/@name'))

    def test_node_path_attribute_dotted(self):
        if False:
            i = 10
            return i + 15
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, '//ReturnStatNode/@value.name')))
        self.assertEqual(['fun'], find_all(t, '//ReturnStatNode/@value.name'))

    def test_node_path_child(self):
        if False:
            for i in range(10):
                print('nop')
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, '//DefNode/ReturnStatNode/NameNode')))
        self.assertEqual(1, len(find_all(t, '//ReturnStatNode/NameNode')))

    def test_node_path_node_predicate(self):
        if False:
            i = 10
            return i + 15
        t = self._build_tree()
        self.assertEqual(0, len(find_all(t, '//DefNode[.//ForInStatNode]')))
        self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode]')))
        self.assertEqual(1, len(find_all(t, '//ReturnStatNode[./NameNode]')))
        self.assertEqual(Nodes.ReturnStatNode, type(find_first(t, '//ReturnStatNode[./NameNode]')))

    def test_node_path_node_predicate_step(self):
        if False:
            i = 10
            return i + 15
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode]')))
        self.assertEqual(8, len(find_all(t, '//DefNode[.//NameNode]//*')))
        self.assertEqual(1, len(find_all(t, '//DefNode[.//NameNode]//ReturnStatNode')))
        self.assertEqual(Nodes.ReturnStatNode, type(find_first(t, '//DefNode[.//NameNode]//ReturnStatNode')))

    def test_node_path_attribute_exists(self):
        if False:
            for i in range(10):
                print('nop')
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, '//NameNode[@name]')))
        self.assertEqual(ExprNodes.NameNode, type(find_first(t, '//NameNode[@name]')))

    def test_node_path_attribute_exists_not(self):
        if False:
            print('Hello World!')
        t = self._build_tree()
        self.assertEqual(0, len(find_all(t, '//NameNode[not(@name)]')))
        self.assertEqual(2, len(find_all(t, '//NameNode[not(@honking)]')))

    def test_node_path_and(self):
        if False:
            print('Hello World!')
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, '//DefNode[.//ReturnStatNode and .//NameNode]')))
        self.assertEqual(0, len(find_all(t, '//NameNode[@honking and @name]')))
        self.assertEqual(0, len(find_all(t, '//NameNode[@name and @honking]')))
        self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode[@name] and @name]')))

    def test_node_path_attribute_string_predicate(self):
        if False:
            while True:
                i = 10
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, "//NameNode[@name = 'decorator']")))

    def test_node_path_recursive_predicate(self):
        if False:
            for i in range(10):
                print('nop')
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode[@name]]')))
        self.assertEqual(1, len(find_all(t, "//DefNode[.//NameNode[@name = 'decorator']]")))
        self.assertEqual(1, len(find_all(t, "//DefNode[.//ReturnStatNode[./NameNode[@name = 'fun']]/NameNode]")))
if __name__ == '__main__':
    unittest.main()