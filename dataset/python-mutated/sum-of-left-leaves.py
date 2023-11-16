class Solution(object):

    def sumOfLeftLeaves(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def sumOfLeftLeavesHelper(root, is_left):
            if False:
                while True:
                    i = 10
            if not root:
                return 0
            if not root.left and (not root.right):
                return root.val if is_left else 0
            return sumOfLeftLeavesHelper(root.left, True) + sumOfLeftLeavesHelper(root.right, False)
        return sumOfLeftLeavesHelper(root, False)