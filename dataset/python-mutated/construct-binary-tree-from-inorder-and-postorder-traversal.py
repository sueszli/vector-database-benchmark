class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def buildTree(self, inorder, postorder):
        if False:
            while True:
                i = 10
        lookup = {}
        for (i, num) in enumerate(inorder):
            lookup[num] = i
        return self.buildTreeRecu(lookup, postorder, inorder, len(postorder), 0, len(inorder))

    def buildTreeRecu(self, lookup, postorder, inorder, post_end, in_start, in_end):
        if False:
            for i in range(10):
                print('nop')
        if in_start == in_end:
            return None
        node = TreeNode(postorder[post_end - 1])
        i = lookup[postorder[post_end - 1]]
        node.left = self.buildTreeRecu(lookup, postorder, inorder, post_end - 1 - (in_end - i - 1), in_start, i)
        node.right = self.buildTreeRecu(lookup, postorder, inorder, post_end - 1, i + 1, in_end)
        return node