class Solution(object):

    def tree2str(self, t):
        if False:
            i = 10
            return i + 15
        '\n        :type t: TreeNode\n        :rtype: str\n        '
        if not t:
            return ''
        s = str(t.val)
        if t.left or t.right:
            s += '(' + self.tree2str(t.left) + ')'
        if t.right:
            s += '(' + self.tree2str(t.right) + ')'
        return s