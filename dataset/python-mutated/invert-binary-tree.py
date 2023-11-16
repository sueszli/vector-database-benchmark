import collections

class Queue(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.data = collections.deque()

    def push(self, x):
        if False:
            return 10
        self.data.append(x)

    def peek(self):
        if False:
            while True:
                i = 10
        return self.data[0]

    def pop(self):
        if False:
            print('Hello World!')
        return self.data.popleft()

    def size(self):
        if False:
            i = 10
            return i + 15
        return len(self.data)

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data) == 0

class Solution(object):

    def invertTree(self, root):
        if False:
            for i in range(10):
                print('nop')
        if root is not None:
            nodes = Queue()
            nodes.push(root)
            while not nodes.empty():
                node = nodes.pop()
                (node.left, node.right) = (node.right, node.left)
                if node.left is not None:
                    nodes.push(node.left)
                if node.right is not None:
                    nodes.push(node.right)
        return root

class Solution2(object):

    def invertTree(self, root):
        if False:
            print('Hello World!')
        if root is not None:
            nodes = []
            nodes.append(root)
            while nodes:
                node = nodes.pop()
                (node.left, node.right) = (node.right, node.left)
                if node.left is not None:
                    nodes.append(node.left)
                if node.right is not None:
                    nodes.append(node.right)
        return root

class Solution3(object):

    def invertTree(self, root):
        if False:
            while True:
                i = 10
        if root is not None:
            (root.left, root.right) = (self.invertTree(root.right), self.invertTree(root.left))
        return root