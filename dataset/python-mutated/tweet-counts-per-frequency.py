import collections
import random

class SkipNode(object):

    def __init__(self, level=0, val=None):
        if False:
            print('Hello World!')
        self.val = val
        self.nexts = [None] * level
        self.prevs = [None] * level

class SkipList(object):
    (P_NUMERATOR, P_DENOMINATOR) = (1, 2)
    MAX_LEVEL = 32

    def __init__(self, end=float('inf'), can_duplicated=False):
        if False:
            i = 10
            return i + 15
        random.seed(0)
        self.__head = SkipNode()
        self.__len = 0
        self.__can_duplicated = can_duplicated
        self.add(end)

    def lower_bound(self, target):
        if False:
            return 10
        return self.__lower_bound(target, self.__find_prev_nodes(target))

    def find(self, target):
        if False:
            return 10
        return self.__find(target, self.__find_prev_nodes(target))

    def add(self, val):
        if False:
            for i in range(10):
                print('nop')
        if not self.__can_duplicated and self.find(val):
            return False
        node = SkipNode(self.__random_level(), val)
        if len(self.__head.nexts) < len(node.nexts):
            self.__head.nexts.extend([None] * (len(node.nexts) - len(self.__head.nexts)))
        prevs = self.__find_prev_nodes(val)
        for i in xrange(len(node.nexts)):
            node.nexts[i] = prevs[i].nexts[i]
            if prevs[i].nexts[i]:
                prevs[i].nexts[i].prevs[i] = node
            prevs[i].nexts[i] = node
            node.prevs[i] = prevs[i]
        self.__len += 1
        return True

    def remove(self, val):
        if False:
            print('Hello World!')
        prevs = self.__find_prev_nodes(val)
        curr = self.__find(val, prevs)
        if not curr:
            return False
        self.__len -= 1
        for i in reversed(xrange(len(curr.nexts))):
            prevs[i].nexts[i] = curr.nexts[i]
            if curr.nexts[i]:
                curr.nexts[i].prevs[i] = prevs[i]
            if not self.__head.nexts[i]:
                self.__head.nexts.pop()
        return True

    def __lower_bound(self, val, prevs):
        if False:
            for i in range(10):
                print('nop')
        if prevs:
            candidate = prevs[0].nexts[0]
            if candidate:
                return candidate
        return None

    def __find(self, val, prevs):
        if False:
            while True:
                i = 10
        candidate = self.__lower_bound(val, prevs)
        if candidate and candidate.val == val:
            return candidate
        return None

    def __find_prev_nodes(self, val):
        if False:
            for i in range(10):
                print('nop')
        prevs = [None] * len(self.__head.nexts)
        curr = self.__head
        for i in reversed(xrange(len(self.__head.nexts))):
            while curr.nexts[i] and curr.nexts[i].val < val:
                curr = curr.nexts[i]
            prevs[i] = curr
        return prevs

    def __random_level(self):
        if False:
            for i in range(10):
                print('nop')
        level = 1
        while random.randint(1, SkipList.P_DENOMINATOR) <= SkipList.P_NUMERATOR and level < SkipList.MAX_LEVEL:
            level += 1
        return level

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__len - 1

    def __str__(self):
        if False:
            return 10
        result = []
        for i in reversed(xrange(len(self.__head.nexts))):
            result.append([])
            curr = self.__head.nexts[i]
            while curr:
                result[-1].append(str(curr.val))
                curr = curr.nexts[i]
        return '\n'.join(map(lambda x: '->'.join(x), result))

class TweetCounts(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__records = collections.defaultdict(lambda : SkipList(can_duplicated=True))
        self.__lookup = {'minute': 60, 'hour': 3600, 'day': 86400}

    def recordTweet(self, tweetName, time):
        if False:
            return 10
        '\n        :type tweetName: str\n        :type time: int\n        :rtype: None\n        '
        self.__records[tweetName].add(time)

    def getTweetCountsPerFrequency(self, freq, tweetName, startTime, endTime):
        if False:
            while True:
                i = 10
        '\n        :type freq: str\n        :type tweetName: str\n        :type startTime: int\n        :type endTime: int\n        :rtype: List[int]\n        '
        delta = self.__lookup[freq]
        result = [0] * ((endTime - startTime) // delta + 1)
        it = self.__records[tweetName].lower_bound(startTime)
        while it is not None and it.val <= endTime:
            result[(it.val - startTime) // delta] += 1
            it = it.nexts[0]
        return result
import bisect

class TweetCounts2(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__records = collections.defaultdict(list)
        self.__lookup = {'minute': 60, 'hour': 3600, 'day': 86400}

    def recordTweet(self, tweetName, time):
        if False:
            print('Hello World!')
        '\n        :type tweetName: str\n        :type time: int\n        :rtype: None\n        '
        bisect.insort(self.__records[tweetName], time)

    def getTweetCountsPerFrequency(self, freq, tweetName, startTime, endTime):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type freq: str\n        :type tweetName: str\n        :type startTime: int\n        :type endTime: int\n        :rtype: List[int]\n        '
        delta = self.__lookup[freq]
        i = startTime
        result = []
        while i <= endTime:
            j = min(i + delta, endTime + 1)
            result.append(bisect.bisect_left(self.__records[tweetName], j) - bisect.bisect_left(self.__records[tweetName], i))
            i += delta
        return result

class TweetCounts3(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__records = collections.defaultdict(list)
        self.__lookup = {'minute': 60, 'hour': 3600, 'day': 86400}

    def recordTweet(self, tweetName, time):
        if False:
            return 10
        '\n        :type tweetName: str\n        :type time: int\n        :rtype: None\n        '
        self.__records[tweetName].append(time)

    def getTweetCountsPerFrequency(self, freq, tweetName, startTime, endTime):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type freq: str\n        :type tweetName: str\n        :type startTime: int\n        :type endTime: int\n        :rtype: List[int]\n        '
        delta = self.__lookup[freq]
        result = [0] * ((endTime - startTime) // delta + 1)
        for t in self.__records[tweetName]:
            if startTime <= t <= endTime:
                result[(t - startTime) // delta] += 1
        return result