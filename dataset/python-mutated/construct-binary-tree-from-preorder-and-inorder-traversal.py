class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def buildTree(self, preorder, inorder):
        if False:
            return 10
        lookup = {}
        for (i, num) in enumerate(inorder):
            lookup[num] = i
        return self.buildTreeRecu(lookup, preorder, inorder, 0, 0, len(inorder))

    def buildTreeRecu(self, lookup, preorder, inorder, pre_start, in_start, in_end):
        if False:
            for i in range(10):
                print('nop')
        if in_start == in_end:
            return None
        node = TreeNode(preorder[pre_start])
        i = lookup[preorder[pre_start]]
        node.left = self.buildTreeRecu(lookup, preorder, inorder, pre_start + 1, in_start, i)
        node.right = self.buildTreeRecu(lookup, preorder, inorder, pre_start + 1 + i - in_start, i + 1, in_end)
        return node

class Solution2(object):

    def buildTree(self, preorder, inorder):
        if False:
            while True:
                i = 10
        '\n        :type preorder: List[int]\n        :type inorder: List[int]\n        :rtype: TreeNode\n        '
        preorder_iterator = iter(preorder)
        inorder_lookup = {n: i for (i, n) in enumerate(inorder)}

        def helper(start, end):
            if False:
                for i in range(10):
                    print('nop')
            if start > end:
                return None
            root_val = next(preorder_iterator)
            root = TreeNode(root_val)
            idx = inorder_lookup[root_val]
            root.left = helper(start, idx - 1)
            root.right = helper(idx + 1, end)
            return root
        return helper(0, len(inorder) - 1)