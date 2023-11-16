class Solution(object):

    def splitBST(self, root, V):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type V: int\n        :rtype: List[TreeNode]\n        '
        if not root:
            return (None, None)
        elif root.val <= V:
            result = self.splitBST(root.right, V)
            root.right = result[0]
            return (root, result[1])
        else:
            result = self.splitBST(root.left, V)
            root.left = result[1]
            return (result[0], root)