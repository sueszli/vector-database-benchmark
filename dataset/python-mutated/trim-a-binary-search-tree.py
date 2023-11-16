class Solution(object):

    def trimBST(self, root, L, R):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :type L: int\n        :type R: int\n        :rtype: TreeNode\n        '
        if not root:
            return None
        if root.val < L:
            return self.trimBST(root.right, L, R)
        if root.val > R:
            return self.trimBST(root.left, L, R)
        (root.left, root.right) = (self.trimBST(root.left, L, R), self.trimBST(root.right, L, R))
        return root