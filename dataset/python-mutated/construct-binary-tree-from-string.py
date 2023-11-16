class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def str2tree(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: TreeNode\n        '

        def str2treeHelper(s, i):
            if False:
                i = 10
                return i + 15
            start = i
            if s[i] == '-':
                i += 1
            while i < len(s) and s[i].isdigit():
                i += 1
            node = TreeNode(int(s[start:i]))
            if i < len(s) and s[i] == '(':
                i += 1
                (node.left, i) = str2treeHelper(s, i)
                i += 1
            if i < len(s) and s[i] == '(':
                i += 1
                (node.right, i) = str2treeHelper(s, i)
                i += 1
            return (node, i)
        return str2treeHelper(s, 0)[0] if s else None