import bisect
MAX_HIGH = int(2000000000.0)
result = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in xrange(1, MAX_HIGH):
    if result[-1] >= MAX_HIGH:
        break
    d1 = result[i] % 10 - 1
    if d1 >= 0:
        result.append(result[i] * 10 + d1)
    d2 = result[i] % 10 + 1
    if d2 <= 9:
        result.append(result[i] * 10 + d2)
result.append(float('inf'))

class Solution(object):

    def countSteppingNumbers(self, low, high):
        if False:
            while True:
                i = 10
        '\n        :type low: int\n        :type high: int\n        :rtype: List[int]\n        '
        lit = bisect.bisect_left(result, low)
        rit = bisect.bisect_right(result, high)
        return result[lit:rit]

class Solution2(object):

    def countSteppingNumbers(self, low, high):
        if False:
            return 10
        '\n        :type low: int\n        :type high: int\n        :rtype: List[int]\n        '
        result = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i in xrange(1, high):
            if result[-1] >= high:
                break
            d1 = result[i] % 10 - 1
            if d1 >= 0:
                result.append(result[i] * 10 + d1)
            d2 = result[i] % 10 + 1
            if d2 <= 9:
                result.append(result[i] * 10 + d2)
        result.append(float('inf'))
        lit = bisect.bisect_left(result, low)
        rit = bisect.bisect_right(result, high)
        return result[lit:rit]