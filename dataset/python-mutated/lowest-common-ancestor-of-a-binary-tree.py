class Solution(object):

    def lowestCommonAncestor(self, root, p, q):
        if False:
            i = 10
            return i + 15
        if root in (None, p, q):
            return root
        (left, right) = [self.lowestCommonAncestor(child, p, q) for child in (root.left, root.right)]
        return root if left and right else left or right