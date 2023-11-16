import random

class SkipNode(object):

    def __init__(self, level=0, num=None):
        if False:
            while True:
                i = 10
        self.num = num
        self.nexts = [None] * level

class Skiplist(object):
    (P_NUMERATOR, P_DENOMINATOR) = (1, 2)
    MAX_LEVEL = 32

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__head = SkipNode()
        self.__len = 0

    def search(self, target):
        if False:
            return 10
        '\n        :type target: int\n        :rtype: bool\n        '
        return True if self.__find(target, self.__find_prev_nodes(target)) else False

    def add(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: None\n        '
        node = SkipNode(self.__random_level(), num)
        if len(self.__head.nexts) < len(node.nexts):
            self.__head.nexts.extend([None] * (len(node.nexts) - len(self.__head.nexts)))
        prevs = self.__find_prev_nodes(num)
        for i in xrange(len(node.nexts)):
            node.nexts[i] = prevs[i].nexts[i]
            prevs[i].nexts[i] = node
        self.__len += 1

    def erase(self, num):
        if False:
            print('Hello World!')
        '\n        :type num: int\n        :rtype: bool\n        '
        prevs = self.__find_prev_nodes(num)
        curr = self.__find(num, prevs)
        if not curr:
            return False
        self.__len -= 1
        for i in reversed(xrange(len(curr.nexts))):
            prevs[i].nexts[i] = curr.nexts[i]
            if not self.__head.nexts[i]:
                self.__head.nexts.pop()
        return True

    def __find(self, num, prevs):
        if False:
            for i in range(10):
                print('nop')
        if prevs:
            candidate = prevs[0].nexts[0]
            if candidate and candidate.num == num:
                return candidate
        return None

    def __find_prev_nodes(self, num):
        if False:
            for i in range(10):
                print('nop')
        prevs = [None] * len(self.__head.nexts)
        curr = self.__head
        for i in reversed(xrange(len(self.__head.nexts))):
            while curr.nexts[i] and curr.nexts[i].num < num:
                curr = curr.nexts[i]
            prevs[i] = curr
        return prevs

    def __random_level(self):
        if False:
            while True:
                i = 10
        level = 1
        while random.randint(1, Skiplist.P_DENOMINATOR) <= Skiplist.P_NUMERATOR and level < Skiplist.MAX_LEVEL:
            level += 1
        return level

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.__len

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for i in reversed(xrange(len(self.__head.nexts))):
            result.append([])
            curr = self.__head.nexts[i]
            while curr:
                result[-1].append(str(curr.num))
                curr = curr.nexts[i]
        return '\n'.join(map(lambda x: '->'.join(x), result))