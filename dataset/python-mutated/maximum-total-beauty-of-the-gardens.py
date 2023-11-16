import bisect

class Solution(object):

    def maximumBeauty(self, flowers, newFlowers, target, full, partial):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type flowers: List[int]\n        :type newFlowers: int\n        :type target: int\n        :type full: int\n        :type partial: int\n        :rtype: int\n        '
        flowers.sort()
        n = bisect.bisect_left(flowers, target)
        (prefix, suffix) = (0, sum((flowers[i] for i in xrange(n))))
        result = left = 0
        for right in xrange(n + 1):
            if right:
                suffix -= flowers[right - 1]
            total = newFlowers - ((n - right) * target - suffix)
            if total < 0:
                continue
            while not (left == right or (left and (total + prefix) // left <= flowers[left])):
                prefix += flowers[left]
                left += 1
            mn = min((total + prefix) // left if left else 0, target - 1)
            result = max(result, mn * partial + (len(flowers) - right) * full)
        return result
import bisect

class Solution2(object):

    def maximumBeauty(self, flowers, newFlowers, target, full, partial):
        if False:
            i = 10
            return i + 15
        '\n        :type flowers: List[int]\n        :type newFlowers: int\n        :type target: int\n        :type full: int\n        :type partial: int\n        :rtype: int\n        '
        flowers.sort()
        n = bisect.bisect_left(flowers, target)
        prefix = [0] * (n + 1)
        for i in xrange(n):
            prefix[i + 1] = prefix[i] + flowers[i]
        result = suffix = 0
        left = n
        for right in reversed(xrange(n + 1)):
            if right != n:
                suffix += flowers[right]
            total = newFlowers - ((n - right) * target - suffix)
            if total < 0:
                continue
            left = min(left, right)
            while not (left == 0 or (prefix[left] - prefix[left - 1]) * left - prefix[left] <= total):
                left -= 1
            mn = min((total + prefix[left]) // left if left else 0, target - 1)
            result = max(result, mn * partial + (len(flowers) - right) * full)
        return result
import bisect

class Solution3(object):

    def maximumBeauty(self, flowers, newFlowers, target, full, partial):
        if False:
            while True:
                i = 10
        '\n        :type flowers: List[int]\n        :type newFlowers: int\n        :type target: int\n        :type full: int\n        :type partial: int\n        :rtype: int\n        '

        def check(prefix, total, x):
            if False:
                print('Hello World!')
            return x and (total + prefix[x]) // x <= prefix[x + 1] - prefix[x]

        def binary_search(prefix, total, left, right):
            if False:
                i = 10
                return i + 15
            while left <= right:
                mid = left + (right - left) // 2
                if check(prefix, total, mid):
                    right = mid - 1
                else:
                    left = mid + 1
            return left
        flowers.sort()
        n = bisect.bisect_left(flowers, target)
        prefix = [0] * (n + 1)
        for i in xrange(n):
            prefix[i + 1] = prefix[i] + flowers[i]
        suffix = sum((flowers[i] for i in xrange(n)))
        result = left = 0
        for right in xrange(n + 1):
            if right:
                suffix -= flowers[right - 1]
            total = newFlowers - ((n - right) * target - suffix)
            if total < 0:
                continue
            left = binary_search(prefix, total, 0, right - 1)
            mn = min((total + prefix[left]) // left if left else 0, target - 1)
            result = max(result, mn * partial + (len(flowers) - right) * full)
        return result
import bisect

class Solution4(object):

    def maximumBeauty(self, flowers, newFlowers, target, full, partial):
        if False:
            while True:
                i = 10
        '\n        :type flowers: List[int]\n        :type newFlowers: int\n        :type target: int\n        :type full: int\n        :type partial: int\n        :rtype: int\n        '

        def check(prefix, total, x):
            if False:
                i = 10
                return i + 15
            return (prefix[x] - prefix[x - 1]) * x - prefix[x] <= total

        def binary_search_right(prefix, total, left, right):
            if False:
                return 10
            while left <= right:
                mid = left + (right - left) // 2
                if not check(prefix, total, mid):
                    right = mid - 1
                else:
                    left = mid + 1
            return right
        flowers.sort()
        n = bisect.bisect_left(flowers, target)
        prefix = [0] * (n + 1)
        for i in xrange(n):
            prefix[i + 1] = prefix[i] + flowers[i]
        result = suffix = 0
        left = n
        for right in reversed(xrange(n + 1)):
            if right != n:
                suffix += flowers[right]
            total = newFlowers - ((n - right) * target - suffix)
            if total < 0:
                break
            left = binary_search_right(prefix, total, 1, right)
            mn = min((total + prefix[left]) // left if left else 0, target - 1)
            result = max(result, mn * partial + (len(flowers) - right) * full)
        return result