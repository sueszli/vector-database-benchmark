class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def lcaDeepestLeaves(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def lcaDeepestLeavesHelper(root):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return (0, None)
            (d1, lca1) = lcaDeepestLeavesHelper(root.left)
            (d2, lca2) = lcaDeepestLeavesHelper(root.right)
            if d1 > d2:
                return (d1 + 1, lca1)
            if d1 < d2:
                return (d2 + 1, lca2)
            return (d1 + 1, root)
        return lcaDeepestLeavesHelper(root)[1]