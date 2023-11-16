import abc
from abc import ABCMeta, abstractmethod

class Node:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self):
        if False:
            return 10
        pass
import operator

class NodeIter(Node):
    ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.div}

    def __init__(self, val):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.left = None
        self.right = None

    def evaluate(self):
        if False:
            print('Hello World!')
        result = [0]
        stk = [(1, (self, result))]
        while stk:
            (step, args) = stk.pop()
            if step == 1:
                (node, ret) = args
                if node.val.isdigit():
                    ret[0] = int(node.val)
                    continue
                (ret1, ret2) = ([0], [0])
                stk.append((2, (node, ret1, ret2, ret)))
                stk.append((1, (node.right, ret2)))
                stk.append((1, (node.left, ret1)))
            elif step == 2:
                (node, ret1, ret2, ret) = args
                ret[0] = NodeIter.ops[node.val](ret1[0], ret2[0])
        return result[0]

class TreeBuilder(object):

    def buildTree(self, postfix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: List[str]\n        :rtype: int\n        '
        stk = []
        for c in postfix:
            if c.isdigit():
                stk.append(NodeIter(c))
            else:
                node = NodeIter(c)
                node.right = stk.pop()
                node.left = stk.pop()
                stk.append(node)
        return stk.pop()

class NodeRecu(Node):
    ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.div}

    def __init__(self, val):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.left = None
        self.right = None

    def evaluate(self):
        if False:
            while True:
                i = 10
        if self.val.isdigit():
            return int(self.val)
        return NodeRecu.ops[self.val](self.left.evaluate(), self.right.evaluate())

class TreeBuilder2(object):

    def buildTree(self, postfix):
        if False:
            return 10
        '\n        :type s: List[str]\n        :rtype: int\n        '
        stk = []
        for c in postfix:
            if c.isdigit():
                stk.append(NodeRecu(c))
            else:
                node = NodeRecu(c)
                node.right = stk.pop()
                node.left = stk.pop()
                stk.append(node)
        return stk.pop()