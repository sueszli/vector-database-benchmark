class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def isEvenOddTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: bool\n        '
        q = [root]
        is_odd = False
        while q:
            new_q = []
            prev = None
            for node in q:
                if is_odd:
                    if node.val % 2 or (prev and prev.val <= node.val):
                        return False
                elif not node.val % 2 or (prev and prev.val >= node.val):
                    return False
                if node.left:
                    new_q.append(node.left)
                if node.right:
                    new_q.append(node.right)
                prev = node
            q = new_q
            is_odd = not is_odd
        return True