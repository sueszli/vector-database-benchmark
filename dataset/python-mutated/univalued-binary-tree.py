class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def isUnivalTree(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: bool\n        '
        s = [root]
        while s:
            node = s.pop()
            if not node:
                continue
            if node.val != root.val:
                return False
            s.append(node.left)
            s.append(node.right)
        return True

class Solution2(object):

    def isUnivalTree(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: bool\n        '
        return (not root.left or (root.left.val == root.val and self.isUnivalTree(root.left))) and (not root.right or (root.right.val == root.val and self.isUnivalTree(root.right)))