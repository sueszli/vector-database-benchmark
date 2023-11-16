class TreeNode(object):

    def __init__(self, val=0, left=None, right=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right

class BSTIterator(object):

    def __init__(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        '
        self.__stk = []
        self.__traversalLeft(root)
        self.__vals = []
        self.__pos = -1

    def hasNext(self):
        if False:
            print('Hello World!')
        '\n        :rtype: bool\n        '
        return self.__pos + 1 != len(self.__vals) or self.__stk

    def next(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        self.__pos += 1
        if self.__pos == len(self.__vals):
            node = self.__stk.pop()
            self.__traversalLeft(node.right)
            self.__vals.append(node.val)
        return self.__vals[self.__pos]

    def hasPrev(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: bool\n        '
        return self.__pos - 1 >= 0

    def prev(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        self.__pos -= 1
        return self.__vals[self.__pos]

    def __traversalLeft(self, node):
        if False:
            for i in range(10):
                print('nop')
        while node is not None:
            self.__stk.append(node)
            node = node.left