class Solution(object):

    def maximumProduct(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        nums.sort()
        total = sum(nums)
        for i in reversed(xrange(len(nums))):
            if nums[i] * (i + 1) - total <= k:
                break
            total -= nums[i]
        (q, r) = divmod(k + total, i + 1)
        return pow(q, i + 1 - r, MOD) * pow(q + 1, r, MOD) * reduce(lambda x, y: x * y % MOD, (nums[j] for j in xrange(i + 1, len(nums))), 1) % MOD
import collections

class Solution2(object):

    def maximumProduct(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        cnt = collections.Counter(nums)
        min_num = min(cnt.iterkeys())
        while k:
            c = min(cnt[min_num], k)
            cnt[min_num] -= c
            cnt[min_num + 1] += c
            if not cnt[min_num]:
                del cnt[min_num]
                min_num += 1
            k -= c
        return reduce(lambda total, x: total * pow(x[0], x[1], MOD) % MOD, cnt.iteritems(), 1)
import heapq

class Solution3(object):

    def maximumProduct(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        min_heap = nums
        heapq.heapify(min_heap)
        while k:
            heapq.heappush(min_heap, heapq.heappop(min_heap) + 1)
            k -= 1
        return reduce(lambda x, y: x * y % MOD, min_heap)