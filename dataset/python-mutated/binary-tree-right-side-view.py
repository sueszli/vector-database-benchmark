class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def rightSideView(self, root):
        if False:
            for i in range(10):
                print('nop')
        result = []
        self.rightSideViewDFS(root, 1, result)
        return result

    def rightSideViewDFS(self, node, depth, result):
        if False:
            i = 10
            return i + 15
        if not node:
            return
        if depth > len(result):
            result.append(node.val)
        self.rightSideViewDFS(node.right, depth + 1, result)
        self.rightSideViewDFS(node.left, depth + 1, result)

class Solution2(object):

    def rightSideView(self, root):
        if False:
            return 10
        if root is None:
            return []
        (result, current) = ([], [root])
        while current:
            next_level = []
            for node in current:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            result.append(node.val)
            current = next_level
        return result