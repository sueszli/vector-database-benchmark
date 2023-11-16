class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def maximumAverageSubtree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: float\n        '

        def maximumAverageSubtreeHelper(root, result):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return [0.0, 0]
            (s1, n1) = maximumAverageSubtreeHelper(root.left, result)
            (s2, n2) = maximumAverageSubtreeHelper(root.right, result)
            s = s1 + s2 + root.val
            n = n1 + n2 + 1
            result[0] = max(result[0], s / n)
            return [s, n]
        result = [0]
        maximumAverageSubtreeHelper(root, result)
        return result[0]