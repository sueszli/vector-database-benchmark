class BitCount(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.__l = 0
        self.__n = n
        self.__count = [0] * n

    def __iadd__(self, num):
        if False:
            for i in range(10):
                print('nop')
        self.__l += 1
        base = 1
        for i in xrange(self.__n):
            if num & base:
                self.__count[i] += 1
            base <<= 1
        return self

    def __isub__(self, num):
        if False:
            print('Hello World!')
        self.__l -= 1
        base = 1
        for i in xrange(self.__n):
            if num & base:
                self.__count[i] -= 1
            base <<= 1
        return self

    def bit_and(self):
        if False:
            for i in range(10):
                print('nop')
        (num, base) = (0, 1)
        for i in xrange(self.__n):
            if self.__count[i] == self.__l:
                num |= base
            base <<= 1
        return num

class Solution(object):

    def closestToTarget(self, arr, target):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :type target: int\n        :rtype: int\n        '
        count = BitCount(max(arr).bit_length())
        (result, left) = (float('inf'), 0)
        for right in xrange(len(arr)):
            count += arr[right]
            while left <= right:
                f = count.bit_and()
                result = min(result, abs(f - target))
                if f >= target:
                    break
                count -= arr[left]
                left += 1
        return result

class Solution2(object):

    def closestToTarget(self, arr, target):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :type target: int\n        :rtype: int\n        '
        (result, dp) = (float('inf'), set())
        for x in arr:
            dp = {x} | {f & x for f in dp}
            for f in dp:
                result = min(result, abs(f - target))
        return result