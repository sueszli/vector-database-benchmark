class Node(object):

    def __init__(self, val, children):
        if False:
            while True:
                i = 10
        self.val = val
        self.children = children

class Solution(object):

    def postorder(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Node\n        :rtype: List[int]\n        '
        if not root:
            return []
        (result, stack) = ([], [root])
        while stack:
            node = stack.pop()
            result.append(node.val)
            for child in node.children:
                if child:
                    stack.append(child)
        return result[::-1]

class Solution2(object):

    def postorder(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Node\n        :rtype: List[int]\n        '

        def dfs(root, result):
            if False:
                for i in range(10):
                    print('nop')
            for child in root.children:
                if child:
                    dfs(child, result)
            result.append(root.val)
        result = []
        if root:
            dfs(root, result)
        return result