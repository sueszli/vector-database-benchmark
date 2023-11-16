class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def goodNodes(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        result = 0
        stk = [(root, root.val)]
        while stk:
            (node, curr_max) = stk.pop()
            if not node:
                continue
            curr_max = max(curr_max, node.val)
            result += int(curr_max <= node.val)
            stk.append((node.right, curr_max))
            stk.append((node.left, curr_max))
        return result

class Solution2(object):

    def goodNodes(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node, curr_max):
            if False:
                print('Hello World!')
            if not node:
                return 0
            curr_max = max(curr_max, node.val)
            return int(curr_max <= node.val) + dfs(node.left, curr_max) + dfs(node.right, curr_max)
        return dfs(root, root.val)