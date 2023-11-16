class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            print('Hello World!')
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def getLonelyNodes(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        result = []
        stk = [root]
        while stk:
            node = stk.pop()
            if not node:
                continue
            if node.left and (not node.right):
                result.append(node.left.val)
            elif node.right and (not node.left):
                result.append(node.right.val)
            stk.append(node.right)
            stk.append(node.left)
        return result

class Solution2(object):

    def getLonelyNodes(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '

        def dfs(node, result):
            if False:
                i = 10
                return i + 15
            if not node:
                return
            if node.left and (not node.right):
                result.append(node.left.val)
            elif node.right and (not node.left):
                result.append(node.right.val)
            dfs(node.left, result)
            dfs(node.right, result)
        result = []
        dfs(root, result)
        return result