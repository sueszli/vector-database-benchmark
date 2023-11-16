class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def increasingBST(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def increasingBSTHelper(root, tail):
            if False:
                while True:
                    i = 10
            if not root:
                return tail
            result = increasingBSTHelper(root.left, root)
            root.left = None
            root.right = increasingBSTHelper(root.right, tail)
            return result
        return increasingBSTHelper(root, None)