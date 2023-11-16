import bisect

class Solution(object):

    def minAbsDifference(self, nums, goal):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type goal: int\n        :rtype: int\n        '
        (mx, mn) = (sum((x for x in nums if x > 0)), sum((x for x in nums if x < 0)))
        if goal > mx:
            return goal - mx
        if goal < mn:
            return mn - goal
        result = abs(goal)
        sums1 = set([0])
        for i in xrange(len(nums) // 2):
            for x in list(sums1):
                if x + nums[i] in sums1:
                    continue
                sums1.add(x + nums[i])
                result = min(result, abs(goal - x - nums[i]))
        sorted_sums1 = sorted(sums1)
        sums2 = set([0])
        for i in xrange(len(nums) // 2, len(nums)):
            for x in list(sums2):
                if x + nums[i] in sums2:
                    continue
                sums2.add(x + nums[i])
                ni = bisect.bisect_left(sorted_sums1, goal - x - nums[i])
                if ni < len(sorted_sums1):
                    result = min(result, abs(goal - x - nums[i] - sorted_sums1[ni]))
                if ni > 0:
                    result = min(result, abs(goal - x - nums[i] - sorted_sums1[ni - 1]))
                if result == 0:
                    return result
        return result