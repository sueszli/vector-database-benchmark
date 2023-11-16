class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class BSTIterator(object):

    def __init__(self, root):
        if False:
            print('Hello World!')
        self.__stk = []
        self.__traversalLeft(root)

    def hasNext(self):
        if False:
            return 10
        return self.__stk

    def next(self):
        if False:
            i = 10
            return i + 15
        node = self.__stk.pop()
        self.__traversalLeft(node.right)
        return node.val

    def __traversalLeft(self, node):
        if False:
            for i in range(10):
                print('nop')
        while node is not None:
            self.__stk.append(node)
            node = node.left