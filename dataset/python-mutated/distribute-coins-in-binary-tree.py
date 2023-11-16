class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def distributeCoins(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(root, result):
            if False:
                print('Hello World!')
            if not root:
                return 0
            (left, right) = (dfs(root.left, result), dfs(root.right, result))
            result[0] += abs(left) + abs(right)
            return root.val + left + right - 1
        result = [0]
        dfs(root, result)
        return result[0]