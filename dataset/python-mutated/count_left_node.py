"""
Write a function count_left_node returns the number of left children in the
tree. For example: the following tree has four left children (the nodes
storing the values 6, 3, 7, and 10):

                    9
                 /                     6         12
              / \\       /               3     8   10      15
                 /                              7                18

    count_left_node = 4

"""
import unittest
from bst import Node
from bst import bst

def count_left_node(root):
    if False:
        i = 10
        return i + 15
    if root is None:
        return 0
    elif root.left is None:
        return count_left_node(root.right)
    else:
        return 1 + count_left_node(root.left) + count_left_node(root.right)
'\n    The tree is created for testing:\n\n                    9\n                 /                     6         12\n              / \\       /               3     8   10      15\n                 /                              7                18\n\n    count_left_node = 4\n\n'

class TestSuite(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tree = bst()
        self.tree.insert(9)
        self.tree.insert(6)
        self.tree.insert(12)
        self.tree.insert(3)
        self.tree.insert(8)
        self.tree.insert(10)
        self.tree.insert(15)
        self.tree.insert(7)
        self.tree.insert(18)

    def test_count_left_node(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(4, count_left_node(self.tree.root))
if __name__ == '__main__':
    unittest.main()