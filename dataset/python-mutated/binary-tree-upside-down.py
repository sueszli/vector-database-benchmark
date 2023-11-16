class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def upsideDownBinaryTree(self, root):
        if False:
            print('Hello World!')
        (p, parent, parent_right) = (root, None, None)
        while p:
            left = p.left
            p.left = parent_right
            parent_right = p.right
            p.right = parent
            parent = p
            p = left
        return parent

class Solution2(object):

    def upsideDownBinaryTree(self, root):
        if False:
            print('Hello World!')
        return self.upsideDownBinaryTreeRecu(root, None)

    def upsideDownBinaryTreeRecu(self, p, parent):
        if False:
            i = 10
            return i + 15
        if p is None:
            return parent
        root = self.upsideDownBinaryTreeRecu(p.left, p)
        if parent:
            p.left = parent.right
        else:
            p.left = None
        p.right = parent
        return root