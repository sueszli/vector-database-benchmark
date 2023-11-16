import itertools

class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def leafSimilar(self, root1, root2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :rtype: bool\n        '

        def dfs(node):
            if False:
                print('Hello World!')
            if not node:
                return
            if not node.left and (not node.right):
                yield node.val
            for i in dfs(node.left):
                yield i
            for i in dfs(node.right):
                yield i
        return all((a == b for (a, b) in itertools.izip_longest(dfs(root1), dfs(root2))))