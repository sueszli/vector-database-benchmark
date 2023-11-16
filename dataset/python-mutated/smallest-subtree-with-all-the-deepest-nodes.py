import collections

class Solution(object):

    def subtreeWithAllDeepest(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '
        Result = collections.namedtuple('Result', ('node', 'depth'))

        def dfs(node):
            if False:
                for i in range(10):
                    print('nop')
            if not node:
                return Result(None, 0)
            (left, right) = (dfs(node.left), dfs(node.right))
            if left.depth > right.depth:
                return Result(left.node, left.depth + 1)
            if left.depth < right.depth:
                return Result(right.node, right.depth + 1)
            return Result(node, left.depth + 1)
        return dfs(root).node