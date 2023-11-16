class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def equalToDescendants(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '

        def iter_dfs(node):
            if False:
                i = 10
                return i + 15
            result = 0
            stk = [(1, [node, [0]])]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (node, ret) = args
                    if not node:
                        continue
                    (ret1, ret2) = ([0], [0])
                    stk.append((2, [node, ret1, ret2, ret]))
                    stk.append((1, [node.right, ret2]))
                    stk.append((1, [node.left, ret1]))
                elif step == 2:
                    (node, ret1, ret2, ret) = args
                    if node.val == ret1[0] + ret2[0]:
                        result += 1
                    ret[0] = ret1[0] + ret2[0] + node.val
            return result
        return iter_dfs(root)

class Solution2(object):

    def equalToDescendants(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '

        def dfs(node, result):
            if False:
                i = 10
                return i + 15
            if not node:
                return 0
            total = dfs(node.left, result) + dfs(node.right, result)
            if node.val == total:
                result[0] += 1
            return total + node.val
        result = [0]
        dfs(root, result)
        return result[0]