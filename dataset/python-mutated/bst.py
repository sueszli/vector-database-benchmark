"""
Implement Binary Search Tree. It has method:
    1. Insert
    2. Search
    3. Size
    4. Traversal (Preorder, Inorder, Postorder)
"""
import unittest

class Node(object):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = data
        self.left = None
        self.right = None

class BST(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.root = None

    def get_root(self):
        if False:
            for i in range(10):
                print('nop')
        return self.root
    '\n        Get the number of elements\n        Using recursion. Complexity O(logN)\n    '

    def size(self):
        if False:
            return 10
        return self.recur_size(self.root)

    def recur_size(self, root):
        if False:
            print('Hello World!')
        if root is None:
            return 0
        else:
            return 1 + self.recur_size(root.left) + self.recur_size(root.right)
    '\n        Search data in bst\n        Using recursion. Complexity O(logN)\n    '

    def search(self, data):
        if False:
            return 10
        return self.recur_search(self.root, data)

    def recur_search(self, root, data):
        if False:
            print('Hello World!')
        if root is None:
            return False
        if root.data == data:
            return True
        elif data > root.data:
            return self.recur_search(root.right, data)
        else:
            return self.recur_search(root.left, data)
    '\n        Insert data in bst\n        Using recursion. Complexity O(logN)\n    '

    def insert(self, data):
        if False:
            print('Hello World!')
        if self.root:
            return self.recur_insert(self.root, data)
        else:
            self.root = Node(data)
            return True

    def recur_insert(self, root, data):
        if False:
            while True:
                i = 10
        if root.data == data:
            return False
        elif data < root.data:
            if root.left:
                return self.recur_insert(root.left, data)
            else:
                root.left = Node(data)
                return True
        elif root.right:
            return self.recur_insert(root.right, data)
        else:
            root.right = Node(data)
            return True
    '\n        Preorder, Postorder, Inorder traversal bst\n    '

    def preorder(self, root):
        if False:
            for i in range(10):
                print('nop')
        if root:
            print(str(root.data), end=' ')
            self.preorder(root.left)
            self.preorder(root.right)

    def inorder(self, root):
        if False:
            return 10
        if root:
            self.inorder(root.left)
            print(str(root.data), end=' ')
            self.inorder(root.right)

    def postorder(self, root):
        if False:
            i = 10
            return i + 15
        if root:
            self.postorder(root.left)
            self.postorder(root.right)
            print(str(root.data), end=' ')
'\n    The tree is created for testing:\n\n                    10\n                 /                     6         15\n              / \\       /               4     9   12      24\n                 /          /                    7         20      30\n                         /\n                       18\n'

class TestSuite(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tree = BST()
        self.tree.insert(10)
        self.tree.insert(15)
        self.tree.insert(6)
        self.tree.insert(4)
        self.tree.insert(9)
        self.tree.insert(12)
        self.tree.insert(24)
        self.tree.insert(7)
        self.tree.insert(20)
        self.tree.insert(30)
        self.tree.insert(18)

    def test_search(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.tree.search(24))
        self.assertFalse(self.tree.search(50))

    def test_size(self):
        if False:
            return 10
        self.assertEqual(11, self.tree.size())
if __name__ == '__main__':
    unittest.main()