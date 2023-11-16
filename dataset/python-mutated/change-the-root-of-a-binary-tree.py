class Node:

    def __init__(self, val):
        if False:
            while True:
                i = 10
        pass

class Solution(object):

    def flipBinaryTree(self, root, leaf):
        if False:
            return 10
        '\n        :type node: Node\n        :rtype: Node\n        '
        (curr, parent) = (leaf, None)
        while True:
            child = curr.parent
            curr.parent = parent
            if curr.left == parent:
                curr.left = None
            else:
                curr.right = None
            if curr == root:
                break
            if curr.left:
                curr.right = curr.left
            curr.left = child
            (curr, parent) = (child, curr)
        return leaf