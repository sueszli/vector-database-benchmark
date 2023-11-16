class Solution(object):

    def findMaximumXOR(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        class Trie(object):

            def __init__(self, bit_length):
                if False:
                    print('Hello World!')
                self.__nodes = []
                self.__new_node()
                self.__bit_length = bit_length

            def __new_node(self):
                if False:
                    i = 10
                    return i + 15
                self.__nodes.append([-1] * 2)
                return len(self.__nodes) - 1

            def insert(self, num):
                if False:
                    return 10
                curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    x = num >> i
                    if self.__nodes[curr][x & 1] == -1:
                        self.__nodes[curr][x & 1] = self.__new_node()
                    curr = self.__nodes[curr][x & 1]

            def query(self, num):
                if False:
                    return 10
                result = curr = 0
                for i in reversed(xrange(self.__bit_length)):
                    result <<= 1
                    x = num >> i
                    if self.__nodes[curr][1 ^ x & 1] != -1:
                        curr = self.__nodes[curr][1 ^ x & 1]
                        result |= 1
                    else:
                        curr = self.__nodes[curr][x & 1]
                return result
        trie = Trie(max(nums).bit_length())
        result = 0
        for num in nums:
            trie.insert(num)
            result = max(result, trie.query(num))
        return result

class Solution2(object):

    def findMaximumXOR(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for i in reversed(xrange(max(nums).bit_length())):
            result <<= 1
            prefixes = set()
            for n in nums:
                prefixes.add(n >> i)
            for p in prefixes:
                if (result | 1) ^ p in prefixes:
                    result |= 1
                    break
        return result