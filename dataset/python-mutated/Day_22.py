class Node:

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.right = self.left = None
        self.data = data

class Solution:

    def insert(self, root, data):
        if False:
            print('Hello World!')
        if root == None:
            return Node(data)
        elif data <= root.data:
            cur = self.insert(root.left, data)
            root.left = cur
        else:
            cur = self.insert(root.right, data)
            root.right = cur
        return root

    def getHeight(self, root):
        if False:
            print('Hello World!')
        if root:
            leftDepth = self.getHeight(root.left)
            rightDepth = self.getHeight(root.right)
            if leftDepth > rightDepth:
                return leftDepth + 1
            else:
                return rightDepth + 1
        else:
            return -1
T = int(input())
myTree = Solution()
root = None
for i in range(T):
    data = int(input())
    root = myTree.insert(root, data)
height = myTree.getHeight(root)
print(height)