class Solution(object):

    def maximumStrongPairXor(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        class Trie(object):

            def __init__(self, bit_length):
                if False:
                    while True:
                        i = 10
                self.__nodes = []
                self.__cnts = []
                self.__new_node()
                self.__bit_length = bit_length

            def __new_node(self):
                if False:
                    return 10
                self.__nodes.append([-1] * 2)
                self.__cnts.append(0)
                return len(self.__nodes) - 1

            def update(self, num, d):
                if False:
                    return 10
                curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    x = num >> i
                    if self.__nodes[curr][x & 1] == -1:
                        self.__nodes[curr][x & 1] = self.__new_node()
                    curr = self.__nodes[curr][x & 1]
                    self.__cnts[curr] += d

            def query(self, num):
                if False:
                    while True:
                        i = 10
                result = curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    result <<= 1
                    x = num >> i
                    if self.__nodes[curr][1 ^ x & 1] != -1 and self.__cnts[self.__nodes[curr][1 ^ x & 1]]:
                        curr = self.__nodes[curr][1 ^ x & 1]
                        result |= 1
                    else:
                        curr = self.__nodes[curr][x & 1]
                return result
        nums.sort()
        trie = Trie(nums[-1].bit_length())
        result = j = 0
        for (i, num) in enumerate(nums):
            trie.update(num, +1)
            while not nums[i] <= 2 * nums[j]:
                trie.update(nums[j], -1)
                j += 1
            result = max(result, trie.query(num))
        return result

class Solution2(object):

    def maximumStrongPairXor(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        class Trie(object):

            def __init__(self, bit_length):
                if False:
                    i = 10
                    return i + 15
                self.__nodes = []
                self.__mins = []
                self.__maxs = []
                self.__new_node()
                self.__bit_length = bit_length

            def __new_node(self):
                if False:
                    return 10
                self.__nodes.append([-1] * 2)
                self.__mins.append(float('inf'))
                self.__maxs.append(float('-inf'))
                return len(self.__nodes) - 1

            def insert(self, num):
                if False:
                    for i in range(10):
                        print('nop')
                curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    x = num >> i
                    if self.__nodes[curr][x & 1] == -1:
                        self.__nodes[curr][x & 1] = self.__new_node()
                    curr = self.__nodes[curr][x & 1]
                    self.__mins[curr] = min(self.__mins[curr], num)
                    self.__maxs[curr] = max(self.__maxs[curr], num)

            def query(self, num):
                if False:
                    while True:
                        i = 10
                result = curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    result <<= 1
                    x = num >> i
                    y = (result | 1) ^ x
                    assert x != y
                    if self.__nodes[curr][y & 1] != -1 and (x > y and num <= 2 * self.__maxs[self.__nodes[curr][y & 1]] or (x < y and self.__mins[self.__nodes[curr][y & 1]] <= 2 * num)):
                        result |= 1
                        curr = self.__nodes[curr][y & 1]
                    else:
                        curr = self.__nodes[curr][1 ^ y & 1]
                return result
        trie = Trie(max(nums).bit_length())
        result = 0
        for num in nums:
            trie.insert(num)
            result = max(result, trie.query(num))
        return result

class Solution3(object):

    def maximumStrongPairXor(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for i in reversed(xrange(max(nums).bit_length())):
            (prefix_min, prefix_max) = ({}, {})
            for x in nums:
                y = x >> i
                if y not in prefix_min:
                    prefix_min[y] = prefix_max[y] = x
                prefix_min[y] = min(prefix_min[y], x)
                prefix_max[y] = max(prefix_max[y], x)
            result <<= 1
            for x in prefix_min.iterkeys():
                y = (result | 1) ^ x
                assert x != y
                if y in prefix_max and prefix_min[max(x, y)] <= 2 * prefix_max[min(x, y)]:
                    result |= 1
                    break
        return result