import collections
import bisect

class TimeMap(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        '
        self.lookup = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        if False:
            while True:
                i = 10
        '\n        :type key: str\n        :type value: str\n        :type timestamp: int\n        :rtype: None\n        '
        self.lookup[key].append((timestamp, value))

    def get(self, key, timestamp):
        if False:
            while True:
                i = 10
        '\n        :type key: str\n        :type timestamp: int\n        :rtype: str\n        '
        A = self.lookup.get(key, None)
        if A is None:
            return ''
        i = bisect.bisect_right(A, (timestamp + 1, 0))
        return A[i - 1][1] if i else ''