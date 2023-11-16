import collections

class MaxStack(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        initialize your data structure here.\n        '
        self.__idx_to_val = collections.defaultdict(int)
        self.__val_to_idxs = collections.defaultdict(list)
        self.__top = None
        self.__max = None

    def push(self, x):
        if False:
            i = 10
            return i + 15
        '\n        :type x: int\n        :rtype: void\n        '
        idx = self.__val_to_idxs[self.__top][-1] + 1 if self.__val_to_idxs else 0
        self.__idx_to_val[idx] = x
        self.__val_to_idxs[x].append(idx)
        self.__top = x
        self.__max = max(self.__max, x)

    def pop(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        val = self.__top
        self.__remove(val)
        return val

    def top(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        return self.__top

    def peekMax(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: int\n        '
        return self.__max

    def popMax(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        val = self.__max
        self.__remove(val)
        return val

    def __remove(self, val):
        if False:
            i = 10
            return i + 15
        idx = self.__val_to_idxs[val][-1]
        self.__val_to_idxs[val].pop()
        if not self.__val_to_idxs[val]:
            del self.__val_to_idxs[val]
        del self.__idx_to_val[idx]
        if val == self.__top:
            self.__top = self.__idx_to_val[max(self.__idx_to_val.keys())] if self.__idx_to_val else None
        if val == self.__max:
            self.__max = max(self.__val_to_idxs.keys()) if self.__val_to_idxs else None