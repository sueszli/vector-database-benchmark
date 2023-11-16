class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def insertIntoBST(self, root, val):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type val: int\n        :rtype: TreeNode\n        '
        (curr, parent) = (root, None)
        while curr:
            parent = curr
            if val <= curr.val:
                curr = curr.left
            else:
                curr = curr.right
        if not parent:
            root = TreeNode(val)
        elif val <= parent.val:
            parent.left = TreeNode(val)
        else:
            parent.right = TreeNode(val)
        return root

class Solution2(object):

    def insertIntoBST(self, root, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :type val: int\n        :rtype: TreeNode\n        '
        if not root:
            root = TreeNode(val)
        elif val <= root.val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root