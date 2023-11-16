class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def addOneRow(self, root, v, d):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type v: int\n        :type d: int\n        :rtype: TreeNode\n        '
        if d in (0, 1):
            node = TreeNode(v)
            if d == 1:
                node.left = root
            else:
                node.right = root
            return node
        if root and d >= 2:
            root.left = self.addOneRow(root.left, v, d - 1 if d > 2 else 1)
            root.right = self.addOneRow(root.right, v, d - 1 if d > 2 else 0)
        return root