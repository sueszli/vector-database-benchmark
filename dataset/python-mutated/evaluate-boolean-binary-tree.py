class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def evaluateTree(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Optional[TreeNode]\n        :rtype: bool\n        '
        INF = float('inf')
        OP = {2: lambda x, y: x or y, 3: lambda x, y: x and y}

        def iter_dfs(root):
            if False:
                return 10
            ret = [0]
            stk = [(1, (root, ret))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (node, ret) = args
                    if node.left == node.right:
                        ret[0] = node.val
                        continue
                    (ret1, ret2) = ([0], [0])
                    stk.append((2, (node, ret1, ret2, ret)))
                    stk.append((1, (node.right, ret2)))
                    stk.append((1, (node.left, ret1)))
                elif step == 2:
                    (node, ret1, ret2, ret) = args
                    ret[0] = OP[node.val](ret1[0], ret2[0])
            return ret[0]
        return iter_dfs(root)

class Solution2(object):

    def evaluateTree(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Optional[TreeNode]\n        :rtype: bool\n        '
        INF = float('inf')
        OP = {2: lambda x, y: x or y, 3: lambda x, y: x and y}

        def dfs(node):
            if False:
                return 10
            if node.left == node.right:
                return node.val
            return OP[node.val](dfs(node.left), dfs(node.right))
        return dfs(root)