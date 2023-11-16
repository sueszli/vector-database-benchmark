import collections

class Solution(object):

    def sequentialDigits(self, low, high):
        if False:
            while True:
                i = 10
        '\n        :type low: int\n        :type high: int\n        :rtype: List[int]\n        '
        result = []
        q = collections.deque(range(1, 9))
        while q:
            num = q.popleft()
            if num > high:
                continue
            if low <= num:
                result.append(num)
            if num % 10 + 1 < 10:
                q.append(num * 10 + num % 10 + 1)
        return result