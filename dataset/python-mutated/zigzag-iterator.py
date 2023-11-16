import collections

class ZigzagIterator(object):

    def __init__(self, v1, v2):
        if False:
            return 10
        '\n        Initialize your q structure here.\n        :type v1: List[int]\n        :type v2: List[int]\n        '
        self.q = collections.deque([(len(v), iter(v)) for v in (v1, v2) if v])

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: int\n        '
        (len, iter) = self.q.popleft()
        if len > 1:
            self.q.append((len - 1, iter))
        return next(iter)

    def hasNext(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: bool\n        '
        return bool(self.q)