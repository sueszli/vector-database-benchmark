class Node(object):

    def __init__(self, val=None, children=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def findRoot(self, tree):
        if False:
            for i in range(10):
                print('nop')
        "\n        :type tree: List['Node']\n        :rtype: 'Node'\n        "
        root = 0
        for node in tree:
            root ^= id(node)
            for child in node.children:
                root ^= id(child)
        for node in tree:
            if id(node) == root:
                return node
        return None

class Solution2(object):

    def findRoot(self, tree):
        if False:
            return 10
        "\n        :type tree: List['Node']\n        :rtype: 'Node'\n        "
        root = 0
        for node in tree:
            root ^= node.val
            for child in node.children:
                root ^= child.val
        for node in tree:
            if node.val == root:
                return node
        return None

class Solution3(object):

    def findRoot(self, tree):
        if False:
            print('Hello World!')
        "\n        :type tree: List['Node']\n        :rtype: 'Node'\n        "
        root = 0
        for node in tree:
            root += node.val - sum((child.val for child in node.children))
        for node in tree:
            if node.val == root:
                return node
        return None