class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def isBalanced(self, root):
        if False:
            return 10

        def getHeight(root):
            if False:
                return 10
            if root is None:
                return 0
            (left_height, right_height) = (getHeight(root.left), getHeight(root.right))
            if left_height < 0 or right_height < 0 or abs(left_height - right_height) > 1:
                return -1
            return max(left_height, right_height) + 1
        return getHeight(root) >= 0