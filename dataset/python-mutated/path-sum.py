class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def hasPathSum(self, root, sum):
        if False:
            i = 10
            return i + 15
        if root is None:
            return False
        if root.left is None and root.right is None and (root.val == sum):
            return True
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)