class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def levelOrder(self, root):
        if False:
            i = 10
            return i + 15
        if root is None:
            return []
        (result, current) = ([], [root])
        while current:
            (next_level, vals) = ([], [])
            for node in current:
                vals.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            current = next_level
            result.append(vals)
        return result