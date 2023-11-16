class Node:

    def __init__(self, info):
        if False:
            i = 10
            return i + 15
        self.info = info
        self.left = None
        self.right = None
        self.level = None

    def __str__(self):
        if False:
            return 10
        return str(self.info)

class BinarySearchTree:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.root = None

    def create(self, val):
        if False:
            return 10
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root
            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break
'\nNode is defined as\nself.left (the left child of the node)\nself.right (the right child of the node)\nself.info (the value of the node)\n'

def inOrder(root):
    if False:
        for i in range(10):
            print('nop')
    if not root:
        return
    inOrder(root.left)
    print(root.info, end=' ')
    inOrder(root.right)
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
postOrder(tree.root)