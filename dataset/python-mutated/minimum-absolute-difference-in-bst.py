class Solution(object):

    def getMinimumDifference(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def inorderTraversal(root, prev, result):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return (result, prev)
            (result, prev) = inorderTraversal(root.left, prev, result)
            if prev:
                result = min(result, root.val - prev.val)
            return inorderTraversal(root.right, root, result)
        return inorderTraversal(root, None, float('inf'))[0]