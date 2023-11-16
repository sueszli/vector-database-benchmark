class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def compareSubtrees(self, root1, root2):
        if False:
            for i in range(10):
                print('nop')
        if root1 == None and root2 == None:
            return True
        elif root1 == None:
            return False
        elif root2 == None:
            return False
        elif root1.val != root2.val:
            return False
        else:
            return self.compareSubtrees(root1.left, root2.right) and self.compareSubtrees(root1.right, root2.left)

    def isSymmetric(self, root) -> bool:
        if False:
            return 10
        if root.left == None and root.right == None:
            return True
        return self.compareSubtrees(root.left, root.right)