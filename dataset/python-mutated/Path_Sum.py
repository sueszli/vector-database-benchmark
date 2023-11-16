class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def hasPathSum(self, root, targetSum) -> bool:
        if False:
            while True:
                i = 10
        if root == None:
            return False
        elif root.left == None and root.right == None:
            return targetSum == root.val
        else:
            return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)