class Solution(object):

    def countOperationsToEmptyArray(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        idxs = range(len(nums))
        idxs.sort(key=lambda x: nums[x])
        return len(idxs) + sum((len(idxs) - (i + 1) for i in xrange(len(idxs) - 1) if idxs[i] > idxs[i + 1]))

class Solution2(object):

    def countOperationsToEmptyArray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        class BIT(object):

            def __init__(self, n):
                if False:
                    return 10
                self.__bit = [0] * (n + 1)

            def add(self, i, val):
                if False:
                    i = 10
                    return i + 15
                i += 1
                while i < len(self.__bit):
                    self.__bit[i] += val
                    i += i & -i

            def query(self, i):
                if False:
                    i = 10
                    return i + 15
                i += 1
                ret = 0
                while i > 0:
                    ret += self.__bit[i]
                    i -= i & -i
                return ret
        bit = BIT(len(nums))
        idxs = range(len(nums))
        idxs.sort(key=lambda x: nums[x])
        result = len(nums)
        prev = -1
        for i in idxs:
            if prev == -1:
                result += i
            elif prev < i:
                result += i - prev - (bit.query(i) - bit.query(prev - 1))
            else:
                result += len(nums) - 1 - bit.query(len(nums) - 1) - (prev - i - (bit.query(prev) - bit.query(i - 1)))
            bit.add(i, 1)
            prev = i
        return result