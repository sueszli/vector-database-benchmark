class Solution(object):

    def rob(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def robHelper(root):
            if False:
                i = 10
                return i + 15
            if not root:
                return (0, 0)
            (left, right) = (robHelper(root.left), robHelper(root.right))
            return (root.val + left[1] + right[1], max(left) + max(right))
        return max(robHelper(root))