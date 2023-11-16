class Solution(object):

    def mergeTrees(self, t1, t2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type t1: TreeNode\n        :type t2: TreeNode\n        :rtype: TreeNode\n        '
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1