class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            print('Hello World!')
        pass

class Solution(object):

    def checkTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: bool\n        '
        return root.val == root.left.val + root.right.val