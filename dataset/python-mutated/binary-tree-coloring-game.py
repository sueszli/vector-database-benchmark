class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def btreeGameWinningMove(self, root, n, x):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type n: int\n        :type x: int\n        :rtype: bool\n        '

        def count(node, x, left_right):
            if False:
                return 10
            if not node:
                return 0
            (left, right) = (count(node.left, x, left_right), count(node.right, x, left_right))
            if node.val == x:
                (left_right[0], left_right[1]) = (left, right)
            return left + right + 1
        left_right = [0, 0]
        count(root, x, left_right)
        blue = max(max(left_right), n - (sum(left_right) + 1))
        return blue > n - blue