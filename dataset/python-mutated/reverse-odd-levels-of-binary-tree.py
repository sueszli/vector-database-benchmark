class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def reverseOddLevels(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Optional[TreeNode]\n        :rtype: Optional[TreeNode]\n        '
        q = [root]
        parity = 0
        while q:
            if parity:
                (left, right) = (0, len(q) - 1)
                while left < right:
                    (q[left].val, q[right].val) = (q[right].val, q[left].val)
                    left += 1
                    right -= 1
            if not q[0].left:
                break
            new_q = []
            for node in q:
                new_q.append(node.left)
                new_q.append(node.right)
            q = new_q
            parity ^= 1
        return root