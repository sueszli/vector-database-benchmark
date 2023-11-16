class Solution(object):

    def printTree(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: List[List[str]]\n        '

        def getWidth(root):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return 0
            return 2 * max(getWidth(root.left), getWidth(root.right)) + 1

        def getHeight(root):
            if False:
                while True:
                    i = 10
            if not root:
                return 0
            return max(getHeight(root.left), getHeight(root.right)) + 1

        def preorderTraversal(root, level, left, right, result):
            if False:
                return 10
            if not root:
                return
            mid = left + (right - left) / 2
            result[level][mid] = str(root.val)
            preorderTraversal(root.left, level + 1, left, mid - 1, result)
            preorderTraversal(root.right, level + 1, mid + 1, right, result)
        (h, w) = (getHeight(root), getWidth(root))
        result = [[''] * w for _ in xrange(h)]
        preorderTraversal(root, 0, 0, w - 1, result)
        return result