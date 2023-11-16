class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def insertIntoMaxTree(self, root, val):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :type val: int\n        :rtype: TreeNode\n        '
        if not root:
            return TreeNode(val)
        if val > root.val:
            node = TreeNode(val)
            node.left = root
            return node
        curr = root
        while curr.right and curr.right.val > val:
            curr = curr.right
        node = TreeNode(val)
        (curr.right, node.left) = (node, curr.right)
        return root