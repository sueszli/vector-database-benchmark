class Solution(object):

    def minOperations(self, nums, target):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '

        def floor_log2_x(x):
            if False:
                print('Hello World!')
            return x.bit_length() - 1
        total = sum(nums)
        if total < target:
            return -1
        cnt = [0] * (floor_log2_x(max(nums)) + 1)
        for x in nums:
            cnt[floor_log2_x(x)] += 1
        result = 0
        for i in reversed(xrange(len(cnt))):
            for _ in xrange(cnt[i]):
                x = 1 << i
                if x <= target:
                    target -= x
                    total -= x
                elif total - x >= target:
                    total -= x
                else:
                    cnt[i - 1] += 2
                    result += 1
        return result

class Solution2(object):

    def minOperations(self, nums, target):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        total = sum(nums)
        if total < target:
            return -1
        nums.sort()
        result = 0
        while target:
            x = nums.pop()
            if x <= target:
                target -= x
                total -= x
            elif total - x >= target:
                total -= x
            else:
                nums.append(x // 2)
                nums.append(x // 2)
                result += 1
        return result
import heapq

class Solution3(object):

    def minOperations(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        total = sum(nums)
        if total < target:
            return -1
        result = 0
        max_heap = [-x for x in nums]
        heapq.heapify(max_heap)
        while target:
            x = -heapq.heappop(max_heap)
            if x <= target:
                target -= x
                total -= x
            elif total - x >= target:
                total -= x
            else:
                heapq.heappush(max_heap, -x // 2)
                heapq.heappush(max_heap, -x // 2)
                result += 1
        return result

class Solution4(object):

    def minOperations(self, nums, target):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '

        def floor_log2_x(x):
            if False:
                return 10
            return x.bit_length() - 1
        if sum(nums) < target:
            return -1
        cnt = [0] * (floor_log2_x(max(nums)) + 1)
        for x in nums:
            cnt[floor_log2_x(x)] += 1
        result = i = 0
        while i < len(cnt):
            if target & 1 << i:
                if not cnt[i]:
                    j = next((j for j in xrange(i, len(cnt)) if cnt[j]))
                    result += j - i
                    j = i
                    cnt[i] -= 1
                    continue
                cnt[i] -= 1
            if i + 1 < len(cnt):
                cnt[i + 1] += cnt[i] // 2
            i += 1
        return result