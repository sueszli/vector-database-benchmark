class Solution(object):

    def convertBST(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def convertBSTHelper(root, cur_sum):
            if False:
                return 10
            if not root:
                return cur_sum
            if root.right:
                cur_sum = convertBSTHelper(root.right, cur_sum)
            cur_sum += root.val
            root.val = cur_sum
            if root.left:
                cur_sum = convertBSTHelper(root.left, cur_sum)
            return cur_sum
        convertBSTHelper(root, 0)
        return root