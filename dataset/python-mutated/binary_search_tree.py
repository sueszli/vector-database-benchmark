class Node(object):

    def __init__(self, data):
        if False:
            return 10
        self.data = data
        self.rightChild = None
        self.leftChild = None

class BinaryTree(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.root = None

    def insert(self, newData):
        if False:
            i = 10
            return i + 15
        leaf = Node(newData)
        if self.root is None:
            self.root = leaf
        else:
            current = self.root
            parent = self.root
            while current is not None:
                parent = current
                if newData < current.data:
                    current = current.leftChild
                else:
                    current = current.rightChild
            if newData < parent.data:
                parent.leftChild = leaf
            else:
                parent.rightChild = leaf

    def delete(self, data):
        if False:
            while True:
                i = 10
        current = self.root
        parent = self.root
        isLeft = False
        if current is None:
            return False
        while current is not None and current.data is not data:
            parent = current
            if data < current.data:
                current = current.leftChild
                isLeft = True
            else:
                current = current.rightChild
                isLeft = False
        if current is None:
            return False
        if current.leftChild is None and current.rightChild is None:
            if current is self.root:
                self.root = None
            elif isLeft:
                parent.leftChild = None
            else:
                parent.rightChild = None
        elif current.rightChild is None:
            if current is self.root:
                self.root = current.leftChild
            elif isLeft:
                parent.leftChild = current.leftChild
            else:
                parent.rightChild = current.leftChild
        elif current.rightChild is None:
            if current is self.root:
                self.root = current.rightChild
            elif isLeft:
                parent.lChild = current.rightChild
            else:
                parent.rightChild = current.rightChild
        else:
            successor = current.rightChild
            successorParent = current
            while successor.leftChild is not None:
                successorParent = successor
                successor = successor.leftChild
            if current is self.root:
                self.root = successor
            elif isLeft:
                parent.leftChild = successor
            else:
                parent.rightChild = successor
            successor.leftChild = current.leftChild
            if successor is not current.rightChild:
                successorParent.leftChild = successor.rightChild
                successor.rightChild = current.rightChild
        return True

    def minNode(self):
        if False:
            print('Hello World!')
        current = self.root
        while current.leftChild is not None:
            current = current.leftChild
        return current.data

    def maxNode(self):
        if False:
            print('Hello World!')
        current = self.root
        while current.rightChild is not None:
            current = current.rightChild
        return current.data

    def printPostOrder(self):
        if False:
            for i in range(10):
                print('nop')
        global postOrder
        postOrder = []

        def PostOrder(node):
            if False:
                while True:
                    i = 10
            if node is not None:
                PostOrder(node.leftChild)
                PostOrder(node.rightChild)
                postOrder.append(node.data)
        PostOrder(self.root)
        return postOrder

    def printInOrder(self):
        if False:
            i = 10
            return i + 15
        global inOrder
        inOrder = []

        def InOrder(node):
            if False:
                print('Hello World!')
            if node is not None:
                InOrder(node.leftChild)
                inOrder.append(node.data)
                InOrder(node.rightChild)
        InOrder(self.root)
        return inOrder

    def printPreOrder(self):
        if False:
            i = 10
            return i + 15
        global preOrder
        preOrder = []

        def PreOrder(node):
            if False:
                while True:
                    i = 10
            if node is not None:
                preOrder.append(node.data)
                PreOrder(node.leftChild)
                PreOrder(node.rightChild)
        PreOrder(self.root)
        return preOrder

    def treeIsEmpty(self):
        if False:
            return 10
        return self.root is None