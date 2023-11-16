class Solution(object):

    def pruneTree(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if not root.left and (not root.right) and (root.val == 0):
            return None
        return root