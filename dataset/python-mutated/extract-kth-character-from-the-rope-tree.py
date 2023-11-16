class RopeTreeNode(object):

    def __init__(self, len=0, val='', left=None, right=None):
        if False:
            while True:
                i = 10
        pass

class Solution(object):

    def getKthCharacter(self, root, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: Optional[RopeTreeNode]\n        :type k: int\n        :rtype: str\n        '
        while root.len:
            l = max(root.left.len, len(root.left.val)) if root.left else 0
            if k <= l:
                root = root.left
            else:
                k -= l
                root = root.right
        return root.val[k - 1]