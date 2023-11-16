class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def bstToGst(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def bstToGstHelper(root, prev):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return root
            bstToGstHelper(root.right, prev)
            root.val += prev[0]
            prev[0] = root.val
            bstToGstHelper(root.left, prev)
            return root
        prev = [0]
        return bstToGstHelper(root, prev)