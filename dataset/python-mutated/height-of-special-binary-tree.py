class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def heightOfTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '
        result = -1
        stk = [(root, 0)]
        while stk:
            (u, d) = stk.pop()
            result = max(result, d)
            if u.right and u.right.left != u:
                stk.append((u.right, d + 1))
            if u.left and u.left.right != u:
                stk.append((u.left, d + 1))
        return result

class Solution2(object):

    def heightOfTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '
        result = -1
        q = [root]
        while q:
            new_q = []
            for u in q:
                if u.left and u.left.right != u:
                    new_q.append(u.left)
                if u.right and u.right.left != u:
                    new_q.append(u.right)
            q = new_q
            result += 1
        return result