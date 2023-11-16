class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        pass
import collections

class Solution(object):

    def minimumFlips(self, root, result):
        if False:
            while True:
                i = 10
        '\n        :type root: Optional[TreeNode]\n        :type result: bool\n        :rtype: int\n        '
        INF = float('inf')
        OP = {2: lambda x, y: x or y, 3: lambda x, y: x and y, 4: lambda x, y: x ^ y, 5: lambda x, y: not x if x is not None else not y}

        def iter_dfs(root, result):
            if False:
                print('Hello World!')
            ret = collections.defaultdict(lambda : INF)
            stk = [(1, (root, ret))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (node, ret) = args
                    if not node:
                        ret[None] = 0
                        continue
                    if node.left == node.right:
                        ret[True] = node.val ^ 1
                        ret[False] = node.val ^ 0
                        continue
                    ret1 = collections.defaultdict(lambda : INF)
                    ret2 = collections.defaultdict(lambda : INF)
                    stk.append((2, (node, ret1, ret2, ret)))
                    stk.append((1, (node.right, ret2)))
                    stk.append((1, (node.left, ret1)))
                elif step == 2:
                    (node, ret1, ret2, ret) = args
                    for (k1, v1) in ret1.iteritems():
                        for (k2, v2) in ret2.iteritems():
                            ret[OP[node.val](k1, k2)] = min(ret[OP[node.val](k1, k2)], v1 + v2)
            return ret[result]
        return iter_dfs(root, result)
import collections

class Solution2(object):

    def minimumFlips(self, root, result):
        if False:
            return 10
        '\n        :type root: Optional[TreeNode]\n        :type result: bool\n        :rtype: int\n        '
        INF = float('inf')
        OP = {2: lambda x, y: x or y, 3: lambda x, y: x and y, 4: lambda x, y: x ^ y, 5: lambda x, y: not x if x is not None else not y}

        def dfs(node):
            if False:
                i = 10
                return i + 15
            if not node:
                return {None: 0}
            if node.left == node.right:
                return {True: node.val ^ 1, False: node.val ^ 0}
            left = dfs(node.left)
            right = dfs(node.right)
            dp = collections.defaultdict(lambda : INF)
            for (k1, v1) in left.iteritems():
                for (k2, v2) in right.iteritems():
                    dp[OP[node.val](k1, k2)] = min(dp[OP[node.val](k1, k2)], v1 + v2)
            return dp
        return dfs(root)[result]