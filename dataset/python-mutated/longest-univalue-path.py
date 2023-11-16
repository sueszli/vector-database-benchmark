class Solution(object):

    def longestUnivaluePath(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        result = [0]

        def dfs(node):
            if False:
                for i in range(10):
                    print('nop')
            if not node:
                return 0
            (left, right) = (dfs(node.left), dfs(node.right))
            left = left + 1 if node.left and node.left.val == node.val else 0
            right = right + 1 if node.right and node.right.val == node.val else 0
            result[0] = max(result[0], left + right)
            return max(left, right)
        dfs(root)
        return result[0]