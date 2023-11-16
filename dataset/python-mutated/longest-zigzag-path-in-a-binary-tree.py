class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def longestZigZag(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node, result):
            if False:
                while True:
                    i = 10
            if not node:
                return [-1, -1]
            (left, right) = (dfs(node.left, result), dfs(node.right, result))
            result[0] = max(result[0], left[1] + 1, right[0] + 1)
            return [left[1] + 1, right[0] + 1]
        result = [0]
        dfs(root, result)
        return result[0]