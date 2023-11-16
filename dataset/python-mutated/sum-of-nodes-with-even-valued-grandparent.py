class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def sumEvenGrandparent(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def sumEvenGrandparentHelper(root, p, gp):
            if False:
                return 10
            return sumEvenGrandparentHelper(root.left, root.val, p) + sumEvenGrandparentHelper(root.right, root.val, p) + (root.val if gp is not None and gp % 2 == 0 else 0) if root else 0
        return sumEvenGrandparentHelper(root, None, None)