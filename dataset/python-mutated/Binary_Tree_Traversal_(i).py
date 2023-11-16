class Node:

    def __init__(self, key):
        if False:
            return 10
        self.val = key
        self.left = None
        self.right = None

class BinaryTreeTraversals:

    def inorder(self, root):
        if False:
            print('Hello World!')
        if root:
            self.inorder(root.left)
            print(root.val, end=' ')
            self.inorder(root.right)

    def preorder(self, root):
        if False:
            return 10
        if root:
            print(root.val, end=' ')
            self.preorder(root.left)
            self.preorder(root.right)

    def postorder(self, root):
        if False:
            print('Hello World!')
        if root:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.val, end=' ')
t = Node(20)
t.left = Node(10)
t.right = Node(30)
t.left.left = Node(5)
t.right.right = Node(35)
b = BinaryTreeTraversals()
b.inorder(t)
print()
b.preorder(t)
print()
b.postorder(t)