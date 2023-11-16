class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def deepestLeavesSum(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        curr = [root]
        while curr:
            (prev, curr) = (curr, [child for p in curr for child in [p.left, p.right] if child])
        return sum((node.val for node in prev))