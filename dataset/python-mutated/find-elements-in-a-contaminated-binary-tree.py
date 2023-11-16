class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class FindElements(object):

    def __init__(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        '

        def dfs(node, v, lookup):
            if False:
                print('Hello World!')
            if not node:
                return
            node.val = v
            lookup.add(v)
            dfs(node.left, 2 * v + 1, lookup)
            dfs(node.right, 2 * v + 2, lookup)
        self.__lookup = set()
        dfs(root, 0, self.__lookup)

    def find(self, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type target: int\n        :rtype: bool\n        '
        return target in self.__lookup