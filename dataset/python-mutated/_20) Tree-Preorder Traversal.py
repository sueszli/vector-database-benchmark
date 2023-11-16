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
            while True:
                i = 10
        return str(self.info)

class BinarySearchTree:

    def __init__(self):
        if False:
            print('Hello World!')
        self.root = None

    def create(self, val):
        if False:
            i = 10
            return i + 15
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

def preOrder(root):
    if False:
        print('Hello World!')
    if not root:
        return
    print(root.info, end=' ')
    preOrder(root.left)
    preOrder(root.right)
tree = BinarySearchTree()
t = int(input())
arr = list(map(int, input().split()))
for i in range(t):
    tree.create(arr[i])
preOrder(tree.root)