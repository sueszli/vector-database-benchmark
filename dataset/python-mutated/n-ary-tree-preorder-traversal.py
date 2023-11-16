class Node(object):

    def __init__(self, val, children):
        if False:
            print('Hello World!')
        self.val = val
        self.children = children

class Solution(object):

    def preorder(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: Node\n        :rtype: List[int]\n        '
        if not root:
            return []
        (result, stack) = ([], [root])
        while stack:
            node = stack.pop()
            result.append(node.val)
            for child in reversed(node.children):
                if child:
                    stack.append(child)
        return result

class Solution2(object):

    def preorder(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: Node\n        :rtype: List[int]\n        '

        def dfs(root, result):
            if False:
                while True:
                    i = 10
            result.append(root.val)
            for child in root.children:
                if child:
                    dfs(child, result)
        result = []
        if root:
            dfs(root, result)
        return result