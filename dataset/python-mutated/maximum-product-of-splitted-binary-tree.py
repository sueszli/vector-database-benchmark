class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def maxProduct(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def dfs(root, total, result):
            if False:
                while True:
                    i = 10
            if not root:
                return 0
            subtotal = dfs(root.left, total, result) + dfs(root.right, total, result) + root.val
            result[0] = max(result[0], subtotal * (total - subtotal))
            return subtotal
        result = [0]
        dfs(root, dfs(root, 0, result), result)
        return result[0] % MOD