class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def isValidBST(self, root):
        if False:
            print('Hello World!')
        (prev, cur) = (None, root)
        while cur:
            if cur.left is None:
                if prev and prev.val >= cur.val:
                    return False
                prev = cur
                cur = cur.right
            else:
                node = cur.left
                while node.right and node.right != cur:
                    node = node.right
                if node.right is None:
                    node.right = cur
                    cur = cur.left
                else:
                    if prev and prev.val >= cur.val:
                        return False
                    node.right = None
                    prev = cur
                    cur = cur.right
        return True

class Solution2(object):

    def isValidBST(self, root):
        if False:
            while True:
                i = 10
        return self.isValidBSTRecu(root, float('-inf'), float('inf'))

    def isValidBSTRecu(self, root, low, high):
        if False:
            print('Hello World!')
        if root is None:
            return True
        return low < root.val and root.val < high and self.isValidBSTRecu(root.left, low, root.val) and self.isValidBSTRecu(root.right, root.val, high)