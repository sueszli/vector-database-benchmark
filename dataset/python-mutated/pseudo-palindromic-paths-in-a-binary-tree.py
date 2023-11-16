class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def pseudoPalindromicPaths(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        result = 0
        stk = [(root, 0)]
        while stk:
            (node, count) = stk.pop()
            if not node:
                continue
            count ^= 1 << node.val - 1
            result += int(node.left == node.right and count & count - 1 == 0)
            stk.append((node.right, count))
            stk.append((node.left, count))
        return result

class Solution2(object):

    def pseudoPalindromicPaths(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node, count):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return 0
            count ^= 1 << node.val - 1
            return int(node.left == node.right and count & count - 1 == 0) + dfs(node.left, count) + dfs(node.right, count)
        return dfs(root, 0)