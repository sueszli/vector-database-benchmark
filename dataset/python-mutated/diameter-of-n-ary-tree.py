class Node(object):

    def __init__(self, val=None, children=None):
        if False:
            return 10
        self.val = val
        self.children = children if children is not None else []

class Solution(object):

    def diameter(self, root):
        if False:
            i = 10
            return i + 15
        "\n        :type root: 'Node'\n        :rtype: int\n        "

        def iter_dfs(root):
            if False:
                i = 10
                return i + 15
            result = [0] * 2
            stk = [(1, (root, result))]
            while stk:
                (step, params) = stk.pop()
                if step == 1:
                    (node, ret) = params
                    for child in reversed(node.children):
                        ret2 = [0] * 2
                        stk.append((2, (ret2, ret)))
                        stk.append((1, (child, ret2)))
                else:
                    (ret2, ret) = params
                    ret[0] = max(ret[0], ret2[0], ret[1] + ret2[1] + 1)
                    ret[1] = max(ret[1], ret2[1] + 1)
            return result
        return iter_dfs(root)[0]

class Solution2(object):

    def diameter(self, root):
        if False:
            return 10
        "\n        :type root: 'Node'\n        :rtype: int\n        "

        def dfs(node):
            if False:
                while True:
                    i = 10
            (max_dia, max_depth) = (0, 0)
            for child in node.children:
                (child_max_dia, child_max_depth) = dfs(child)
                max_dia = max(max_dia, child_max_dia, max_depth + child_max_depth + 1)
                max_depth = max(max_depth, child_max_depth + 1)
            return (max_dia, max_depth)
        return dfs(root)[0]