class Solution(object):

    def maxPower(self, stations, r, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type stations: List[int]\n        :type r: int\n        :type k: int\n        :rtype: int\n        '

        def min_power():
            if False:
                for i in range(10):
                    print('nop')
            mn = float('inf')
            curr = sum((stations[i] for i in xrange(r)))
            for i in xrange(len(stations)):
                if i + r < len(stations):
                    curr += stations[i + r]
                if i >= r + 1:
                    curr -= stations[i - (r + 1)]
                mn = min(mn, curr)
            return mn

        def check(target):
            if False:
                while True:
                    i = 10
            arr = stations[:]
            curr = sum((arr[i] for i in xrange(r)))
            cnt = k
            for i in xrange(len(arr)):
                if i + r < len(arr):
                    curr += arr[i + r]
                if i >= r + 1:
                    curr -= arr[i - (r + 1)]
                if curr >= target:
                    continue
                diff = target - curr
                if diff > cnt:
                    return False
                cnt -= diff
                curr += diff
                if i + r < len(arr):
                    arr[i + r] += diff
            return True
        mn = min_power()
        (left, right) = (mn, mn + k)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right