class Node:

    def __init__(self, key):
        if False:
            print('Hello World!')
        self.val = key
        self.left = None
        self.right = None

class BinaryTree:

    def height(self, root):
        if False:
            for i in range(10):
                print('nop')
        if root == None:
            return 0
        else:
            return max(self.height(root.left), self.height(root.right)) + 1
t = Node(20)
t.left = Node(10)
t.right = Node(30)
t.left.left = Node(5)
t.right.right = Node(35)
b = BinaryTree()
print(b.height(t))