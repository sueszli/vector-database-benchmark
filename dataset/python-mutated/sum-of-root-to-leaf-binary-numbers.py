class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def sumRootToLeaf(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        M = 10 ** 9 + 7

        def sumRootToLeafHelper(root, val):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return 0
            val = (val * 2 + root.val) % M
            if not root.left and (not root.right):
                return val
            return (sumRootToLeafHelper(root.left, val) + sumRootToLeafHelper(root.right, val)) % M
        return sumRootToLeafHelper(root, 0)