class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def constructMaximumBinaryTree(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: TreeNode\n        '
        nodeStack = []
        for num in nums:
            node = TreeNode(num)
            while nodeStack and num > nodeStack[-1].val:
                node.left = nodeStack.pop()
            if nodeStack:
                nodeStack[-1].right = node
            nodeStack.append(node)
        return nodeStack[0]