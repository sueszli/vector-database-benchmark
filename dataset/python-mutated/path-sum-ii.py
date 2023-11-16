class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def pathSum(self, root, sum):
        if False:
            while True:
                i = 10
        return self.pathSumRecu([], [], root, sum)

    def pathSumRecu(self, result, cur, root, sum):
        if False:
            i = 10
            return i + 15
        if root is None:
            return result
        if root.left is None and root.right is None and (root.val == sum):
            result.append(cur + [root.val])
            return result
        cur.append(root.val)
        self.pathSumRecu(result, cur, root.left, sum - root.val)
        self.pathSumRecu(result, cur, root.right, sum - root.val)
        cur.pop()
        return result