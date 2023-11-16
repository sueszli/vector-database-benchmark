import collections

class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def verticalTraversal(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: List[List[int]]\n        '

        def dfs(node, lookup, x, y):
            if False:
                i = 10
                return i + 15
            if not node:
                return
            lookup[x][y].append(node)
            dfs(node.left, lookup, x - 1, y + 1)
            dfs(node.right, lookup, x + 1, y + 1)
        lookup = collections.defaultdict(lambda : collections.defaultdict(list))
        dfs(root, lookup, 0, 0)
        result = []
        for x in sorted(lookup):
            report = []
            for y in sorted(lookup[x]):
                report.extend(sorted((node.val for node in lookup[x][y])))
            result.append(report)
        return result