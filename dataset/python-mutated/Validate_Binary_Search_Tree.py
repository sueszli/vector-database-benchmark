class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def isValidated(self, node, lower, upper):
        if False:
            for i in range(10):
                print('nop')
        if not node:
            return True
        elif node.val <= lower or node.val >= upper:
            return False
        return self.isValidated(node.left, lower, node.val) and self.isValidated(node.right, node.val, upper)

    def isValidBST(self, root) -> bool:
        if False:
            print('Hello World!')
        return self.isValidated(root, -float('inf'), float('inf'))