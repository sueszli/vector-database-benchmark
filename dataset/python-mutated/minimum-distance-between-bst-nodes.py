class Solution(object):

    def minDiffInBST(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node):
            if False:
                return 10
            if not node:
                return
            dfs(node.left)
            self.result = min(self.result, node.val - self.prev)
            self.prev = node.val
            dfs(node.right)
        self.prev = float('-inf')
        self.result = float('inf')
        dfs(root)
        return self.result