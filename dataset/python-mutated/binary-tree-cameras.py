class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def minCameraCover(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: int\n        '
        (UNCOVERED, COVERED, CAMERA) = range(3)

        def dfs(root, result):
            if False:
                print('Hello World!')
            left = dfs(root.left, result) if root.left else COVERED
            right = dfs(root.right, result) if root.right else COVERED
            if left == UNCOVERED or right == UNCOVERED:
                result[0] += 1
                return CAMERA
            if left == CAMERA or right == CAMERA:
                return COVERED
            return UNCOVERED
        result = [0]
        if dfs(root, result) == UNCOVERED:
            result[0] += 1
        return result[0]