class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def replaceValueInTree(self, root):
        if False:
            return 10
        '\n        :type root: Optional[TreeNode]\n        :rtype: Optional[TreeNode]\n        '
        q = [(root, root.val)]
        while q:
            new_q = []
            total = sum((node.val for (node, _) in q))
            for (node, x) in q:
                node.val = total - x
                x = (node.left.val if node.left else 0) + (node.right.val if node.right else 0)
                if node.left:
                    new_q.append((node.left, x))
                if node.right:
                    new_q.append((node.right, x))
            q = new_q
        return root