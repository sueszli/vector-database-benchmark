class Node(object):

    def __init__(self, val, children):
        if False:
            return 10
        self.val = val
        self.children = children

class Solution(object):

    def maxDepth(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Node\n        :rtype: int\n        '
        if not root:
            return 0
        depth = 0
        for child in root.children:
            depth = max(depth, self.maxDepth(child))
        return 1 + depth