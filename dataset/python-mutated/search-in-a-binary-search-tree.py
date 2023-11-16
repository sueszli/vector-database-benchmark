class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def searchBST(self, root, val):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type val: int\n        :rtype: TreeNode\n        '
        while root and val != root.val:
            if val < root.val:
                root = root.left
            else:
                root = root.right
        return root