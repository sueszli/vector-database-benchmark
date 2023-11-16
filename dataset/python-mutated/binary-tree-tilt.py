class Solution(object):

    def findTilt(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def postOrderTraverse(root, tilt):
            if False:
                while True:
                    i = 10
            if not root:
                return (0, tilt)
            (left, tilt) = postOrderTraverse(root.left, tilt)
            (right, tilt) = postOrderTraverse(root.right, tilt)
            tilt += abs(left - right)
            return (left + right + root.val, tilt)
        return postOrderTraverse(root, 0)[1]