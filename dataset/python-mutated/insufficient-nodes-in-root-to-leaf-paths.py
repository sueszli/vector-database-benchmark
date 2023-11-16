class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def sufficientSubset(self, root, limit):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type limit: int\n        :rtype: TreeNode\n        '
        if not root:
            return None
        if not root.left and (not root.right):
            return None if root.val < limit else root
        root.left = self.sufficientSubset(root.left, limit - root.val)
        root.right = self.sufficientSubset(root.right, limit - root.val)
        if not root.left and (not root.right):
            return None
        return root