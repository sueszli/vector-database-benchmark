import sys

class Node:

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.right = self.left = None
        self.data = data

class Solution:

    def insert(self, root, data):
        if False:
            return 10
        if root == None:
            return Node(data)
        elif data <= root.data:
            cur = self.insert(root.left, data)
            root.left = cur
        else:
            cur = self.insert(root.right, data)
            root.right = cur
        return root

    def levelOrder(self, root):
        if False:
            for i in range(10):
                print('nop')
        from collections import deque
        if root:
            queue = deque([root])
        while queue:
            node = queue.popleft()
            print(node.data, end=' ')
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
T = int(input())
myTree = Solution()
root = None
for i in range(T):
    data = int(input())
    root = myTree.insert(root, data)
myTree.levelOrder(root)