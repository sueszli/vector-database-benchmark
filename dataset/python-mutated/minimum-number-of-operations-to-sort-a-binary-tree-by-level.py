class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def minimumOperations(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Optional[TreeNode]\n        :rtype: int\n        '
        result = 0
        q = [root]
        while q:
            new_q = []
            for node in q:
                if node.left:
                    new_q.append(node.left)
                if node.right:
                    new_q.append(node.right)
            idx = range(len(q))
            idx.sort(key=lambda x: q[x].val)
            for i in xrange(len(q)):
                while idx[i] != i:
                    (idx[idx[i]], idx[i]) = (idx[i], idx[idx[i]])
                    result += 1
            q = new_q
        return result