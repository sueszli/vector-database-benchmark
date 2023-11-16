class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        if False:
            print('Hello World!')
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def levelTraverse(self, node, level):
        if False:
            for i in range(10):
                print('nop')
        if level == len(self.levels):
            self.levels.append([])
        self.levels[level].append(node.val)
        if node.left:
            self.levelTraverse(node.left, level + 1)
        if node.right:
            self.levelTraverse(node.right, level + 1)

    def levelOrder(self, root):
        if False:
            for i in range(10):
                print('nop')
        if not root:
            return []
        self.levels = []
        self.levelTraverse(root, 0)
        return self.levels