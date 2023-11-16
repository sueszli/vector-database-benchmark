class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def sumNumbers(self, root):
        if False:
            i = 10
            return i + 15
        return self.sumNumbersRecu(root, 0)

    def sumNumbersRecu(self, root, num):
        if False:
            print('Hello World!')
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return num * 10 + root.val
        return self.sumNumbersRecu(root.left, num * 10 + root.val) + self.sumNumbersRecu(root.right, num * 10 + root.val)