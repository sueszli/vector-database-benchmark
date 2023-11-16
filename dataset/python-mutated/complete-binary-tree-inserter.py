class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

class CBTInserter(object):

    def __init__(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        '
        self.__tree = [root]
        for i in self.__tree:
            if i.left:
                self.__tree.append(i.left)
            if i.right:
                self.__tree.append(i.right)

    def insert(self, v):
        if False:
            print('Hello World!')
        '\n        :type v: int\n        :rtype: int\n        '
        n = len(self.__tree)
        self.__tree.append(TreeNode(v))
        if n % 2:
            self.__tree[(n - 1) // 2].left = self.__tree[-1]
        else:
            self.__tree[(n - 1) // 2].right = self.__tree[-1]
        return self.__tree[(n - 1) // 2].val

    def get_root(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: TreeNode\n        '
        return self.__tree[0]