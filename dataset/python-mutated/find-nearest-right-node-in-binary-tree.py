class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def findNeartestRightNode(self, root, u):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :type u: TreeNode\n        :rtype: TreeNode\n        '
        q = [root]
        while q:
            new_q = []
            for (i, node) in enumerate(q):
                if node == u:
                    return q[i + 1] if i + 1 < len(q) else None
                if node.left:
                    new_q.append(node.left)
                if node.right:
                    new_q.append(node.right)
            q = new_q
        return None