class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def maxAncestorDiff(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        result = 0
        stack = [(root, 0, float('inf'))]
        while stack:
            (node, mx, mn) = stack.pop()
            if not node:
                continue
            result = max(result, mx - node.val, node.val - mn)
            mx = max(mx, node.val)
            mn = min(mn, node.val)
            stack.append((node.left, mx, mn))
            stack.append((node.right, mx, mn))
        return result

class Solution2(object):

    def maxAncestorDiff(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def maxAncestorDiffHelper(node, mx, mn):
            if False:
                while True:
                    i = 10
            if not node:
                return 0
            result = max(mx - node.val, node.val - mn)
            mx = max(mx, node.val)
            mn = min(mn, node.val)
            result = max(result, maxAncestorDiffHelper(node.left, mx, mn))
            result = max(result, maxAncestorDiffHelper(node.right, mx, mn))
            return result
        return maxAncestorDiffHelper(root, 0, float('inf'))