class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def rangeSumBST(self, root, L, R):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :type L: int\n        :type R: int\n        :rtype: int\n        '
        result = 0
        s = [root]
        while s:
            node = s.pop()
            if node:
                if L <= node.val <= R:
                    result += node.val
                if L < node.val:
                    s.append(node.left)
                if node.val < R:
                    s.append(node.right)
        return result