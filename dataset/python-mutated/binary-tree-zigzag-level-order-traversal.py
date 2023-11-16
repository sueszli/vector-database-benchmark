class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def zigzagLevelOrder(self, root):
        if False:
            while True:
                i = 10
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
            result.append(vals[::-1] if len(result) % 2 else vals)
            current = next_level
        return result