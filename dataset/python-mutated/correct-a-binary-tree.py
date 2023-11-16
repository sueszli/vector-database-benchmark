class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            print('Hello World!')
        pass

class Solution(object):

    def correctBinaryTree(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '
        q = {root: None}
        while q:
            new_q = {}
            for (node, parent) in q.iteritems():
                if node.right in q:
                    if parent.left == node:
                        parent.left = None
                    else:
                        parent.right = None
                    return root
                if node.left:
                    new_q[node.left] = node
                if node.right:
                    new_q[node.right] = node
            q = new_q