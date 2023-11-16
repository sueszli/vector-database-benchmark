class ZigZagIterator:

    def __init__(self, v1, v2):
        if False:
            print('Hello World!')
        '\n        Initialize your data structure here.\n        :type v1: List[int]\n        :type v2: List[int]\n        '
        self.queue = [_ for _ in (v1, v2) if _]
        print(self.queue)

    def next(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        v = self.queue.pop(0)
        ret = v.pop(0)
        if v:
            self.queue.append(v)
        return ret

    def has_next(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: bool\n        '
        if self.queue:
            return True
        return False
l1 = [1, 2]
l2 = [3, 4, 5, 6]
it = ZigZagIterator(l1, l2)
while it.has_next():
    print(it.next())