class BSTIterator:

    def __init__(self, root):
        if False:
            return 10
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def has_next(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.stack)

    def next(self):
        if False:
            print('Hello World!')
        node = self.stack.pop()
        tmp = node
        if tmp.right:
            tmp = tmp.right
            while tmp:
                self.stack.append(tmp)
                tmp = tmp.left
        return node.val