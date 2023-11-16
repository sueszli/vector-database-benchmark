class Solution(object):

    def isSubtree(self, s, t):
        if False:
            return 10
        '\n        :type s: TreeNode\n        :type t: TreeNode\n        :rtype: bool\n        '

        def isSame(x, y):
            if False:
                print('Hello World!')
            if not x and (not y):
                return True
            if not x or not y:
                return False
            return x.val == y.val and isSame(x.left, y.left) and isSame(x.right, y.right)

        def preOrderTraverse(s, t):
            if False:
                print('Hello World!')
            return s != None and (isSame(s, t) or preOrderTraverse(s.left, t) or preOrderTraverse(s.right, t))
        return preOrderTraverse(s, t)