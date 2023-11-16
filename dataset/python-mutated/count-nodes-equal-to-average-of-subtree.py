class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def averageOfSubtree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '

        def iter_dfs(root):
            if False:
                while True:
                    i = 10
            result = 0
            stk = [(1, (root, [0] * 2))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (node, ret) = args
                    if not node:
                        continue
                    (ret1, ret2) = ([0] * 2, [0] * 2)
                    stk.append((2, (node, ret1, ret2, ret)))
                    stk.append((1, (node.right, ret2)))
                    stk.append((1, (node.left, ret1)))
                elif step == 2:
                    (node, ret1, ret2, ret) = args
                    ret[0] = ret1[0] + ret2[0] + node.val
                    ret[1] = ret1[1] + ret2[1] + 1
                    result += int(ret[0] // ret[1] == node.val)
            return result
        return iter_dfs(root)

class Solution2(object):

    def averageOfSubtree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '

        def dfs(node):
            if False:
                i = 10
                return i + 15
            if not node:
                return [0] * 3
            left = dfs(node.left)
            right = dfs(node.right)
            return [left[0] + right[0] + node.val, left[1] + right[1] + 1, left[2] + right[2] + int((left[0] + right[0] + node.val) // (left[1] + right[1] + 1) == node.val)]
        return dfs(root)[2]