class Node(object):

    def __init__(self, val, children):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.children = children

class Solution(object):

    def levelOrder(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: Node\n        :rtype: List[List[int]]\n        '
        if not root:
            return []
        (result, q) = ([], [root])
        while q:
            result.append([node.val for node in q])
            q = [child for node in q for child in node.children if child]
        return result