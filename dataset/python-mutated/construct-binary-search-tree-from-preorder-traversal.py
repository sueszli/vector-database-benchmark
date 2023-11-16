class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def bstFromPreorder(self, preorder):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type preorder: List[int]\n        :rtype: TreeNode\n        '

        def bstFromPreorderHelper(preorder, left, right, index):
            if False:
                return 10
            if index[0] == len(preorder) or preorder[index[0]] < left or preorder[index[0]] > right:
                return None
            root = TreeNode(preorder[index[0]])
            index[0] += 1
            root.left = bstFromPreorderHelper(preorder, left, root.val, index)
            root.right = bstFromPreorderHelper(preorder, root.val, right, index)
            return root
        return bstFromPreorderHelper(preorder, float('-inf'), float('inf'), [0])