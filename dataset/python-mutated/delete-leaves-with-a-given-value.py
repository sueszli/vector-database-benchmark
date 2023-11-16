class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def removeLeafNodes(self, root, target):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type target: int\n        :rtype: TreeNode\n        '
        if not root:
            return None
        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)
        return None if root.left == root.right and root.val == target else root