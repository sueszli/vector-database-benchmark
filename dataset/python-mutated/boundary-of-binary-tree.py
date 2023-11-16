class Solution(object):

    def boundaryOfBinaryTree(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '

        def leftBoundary(root, nodes):
            if False:
                i = 10
                return i + 15
            if not root or (not root.left and (not root.right)):
                return
            nodes.append(root.val)
            if not root.left:
                leftBoundary(root.right, nodes)
            else:
                leftBoundary(root.left, nodes)

        def rightBoundary(root, nodes):
            if False:
                return 10
            if not root or (not root.left and (not root.right)):
                return
            if not root.right:
                rightBoundary(root.left, nodes)
            else:
                rightBoundary(root.right, nodes)
            nodes.append(root.val)

        def leaves(root, nodes):
            if False:
                while True:
                    i = 10
            if not root:
                return
            if not root.left and (not root.right):
                nodes.append(root.val)
                return
            leaves(root.left, nodes)
            leaves(root.right, nodes)
        if not root:
            return []
        nodes = [root.val]
        leftBoundary(root.left, nodes)
        leaves(root.left, nodes)
        leaves(root.right, nodes)
        rightBoundary(root.right, nodes)
        return nodes