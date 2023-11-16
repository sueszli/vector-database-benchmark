import collections

class Solution(object):

    def countPairs(self, nums, low, high):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type low: int\n        :type high: int\n        :rtype: int\n        '

        def count(nums, x):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            dp = collections.Counter(nums)
            while x:
                if x & 1:
                    result += sum((dp[x ^ 1 ^ k] * dp[k] for k in dp.iterkeys())) // 2
                dp = collections.Counter({k >> 1: dp[k] + dp[k ^ 1] for k in dp.iterkeys()})
                x >>= 1
            return result
        return count(nums, high + 1) - count(nums, low)

class Trie(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__root = {}

    def insert(self, num):
        if False:
            while True:
                i = 10
        node = self.__root
        for i in reversed(xrange(32)):
            curr = num >> i & 1
            if curr not in node:
                node[curr] = {'_count': 0}
            node = node[curr]
            node['_count'] += 1

    def query(self, num, limit):
        if False:
            i = 10
            return i + 15
        (node, result) = (self.__root, 0)
        for i in reversed(xrange(32)):
            curr = num >> i & 1
            bit = limit >> i & 1
            if bit:
                if curr in node:
                    result += node[0 ^ curr]['_count']
            if bit ^ curr not in node:
                break
            node = node[bit ^ curr]
        return result

class Solution2(object):

    def countPairs(self, nums, low, high):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type low: int\n        :type high: int\n        :rtype: int\n        '
        result = 0
        trie = Trie()
        for x in nums:
            result += trie.query(x, high + 1) - trie.query(x, low)
            trie.insert(x)
        return result