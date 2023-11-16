class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def maxSumBST(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        result = 0
        stk = [[root, None, []]]
        while stk:
            (node, tmp, ret) = stk.pop()
            if tmp:
                (lvalid, lsum, lmin, lmax) = tmp[0]
                (rvalid, rsum, rmin, rmax) = tmp[1]
                if lvalid and rvalid and (lmax < node.val < rmin):
                    total = lsum + node.val + rsum
                    result = max(result, total)
                    ret[:] = [True, total, min(lmin, node.val), max(node.val, rmax)]
                    continue
                ret[:] = [False, 0, 0, 0]
                continue
            if not node:
                ret[:] = [True, 0, float('inf'), float('-inf')]
                continue
            new_tmp = [[], []]
            stk.append([node, new_tmp, ret])
            stk.append([node.right, None, new_tmp[1]])
            stk.append([node.left, None, new_tmp[0]])
        return result

class Solution2(object):

    def maxSumBST(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node, result):
            if False:
                i = 10
                return i + 15
            if not node:
                return (True, 0, float('inf'), float('-inf'))
            (lvalid, lsum, lmin, lmax) = dfs(node.left, result)
            (rvalid, rsum, rmin, rmax) = dfs(node.right, result)
            if lvalid and rvalid and (lmax < node.val < rmin):
                total = lsum + node.val + rsum
                result[0] = max(result[0], total)
                return (True, total, min(lmin, node.val), max(node.val, rmax))
            return (False, 0, 0, 0)
        result = [0]
        dfs(root, result)
        return result[0]