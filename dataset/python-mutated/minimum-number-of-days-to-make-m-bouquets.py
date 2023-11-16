class Solution(object):

    def minDays(self, bloomDay, m, k):
        if False:
            print('Hello World!')
        '\n        :type bloomDay: List[int]\n        :type m: int\n        :type k: int\n        :rtype: int\n        '

        def check(bloomDay, m, k, x):
            if False:
                return 10
            result = count = 0
            for d in bloomDay:
                count = count + 1 if d <= x else 0
                if count == k:
                    count = 0
                    result += 1
                    if result == m:
                        break
            return result >= m
        if m * k > len(bloomDay):
            return -1
        (left, right) = (1, max(bloomDay))
        while left <= right:
            mid = left + (right - left) // 2
            if check(bloomDay, m, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left