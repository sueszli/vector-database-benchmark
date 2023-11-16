class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            return 10
        pass

class Solution(object):

    def findDistance(self, root, p, q):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :type p: int\n        :type q: int\n        :rtype: int\n        '

        def iter_dfs(root, p, q):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            dist = [-1]
            stk = [(1, [root, dist])]
            while stk:
                (step, params) = stk.pop()
                if step == 1:
                    (node, ret) = params
                    if not node:
                        continue
                    (ret1, ret2) = ([-1], [-1])
                    stk.append((2, [node, ret1, ret2, ret]))
                    stk.append((1, [node.right, ret2]))
                    stk.append((1, [node.left, ret1]))
                elif step == 2:
                    (node, ret1, ret2, ret) = params
                    if node.val in (p, q):
                        if ret1[0] == ret2[0] == -1:
                            ret[0] = 0
                        else:
                            result = ret1[0] + 1 if ret1[0] != -1 else ret2[0] + 1
                    elif ret1[0] != -1 and ret2[0] != -1:
                        result = ret1[0] + ret2[0] + 2
                    elif ret1[0] != -1:
                        ret[0] = ret1[0] + 1
                    elif ret2[0] != -1:
                        ret[0] = ret2[0] + 1
            return result
        return iter_dfs(root, p, q)

class Solution2(object):

    def findDistance(self, root, p, q):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :type p: int\n        :type q: int\n        :rtype: int\n        '

        def dfs(node, p, q, result):
            if False:
                i = 10
                return i + 15
            if not node:
                return -1
            left = dfs(node.left, p, q, result)
            right = dfs(node.right, p, q, result)
            if node.val in (p, q):
                if left == right == -1:
                    return 0
                result[0] = left + 1 if left != -1 else right + 1
            if left != -1 and right != -1:
                result[0] = left + right + 2
            elif left != -1:
                return left + 1
            elif right != -1:
                return right + 1
            return -1
        result = [0]
        dfs(root, p, q, result)
        return result[0]