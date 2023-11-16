import collections

class Solution(object):

    def minChanges(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def one_are_not_from_nums(nums, cnts):
            if False:
                while True:
                    i = 10
            mxs = [cnts[i].most_common(1)[0][1] for i in xrange(k)]
            return len(nums) - (sum(mxs) - min(mxs))

        def all_are_from_nums(nums, cnts):
            if False:
                while True:
                    i = 10
            dp = {0: 0}
            for cnt in cnts:
                new_dp = collections.defaultdict(int)
                for x in dp.iterkeys():
                    for y in cnt.iterkeys():
                        new_dp[x ^ y] = max(new_dp[x ^ y], dp[x] + cnt[y])
                dp = new_dp
            return len(nums) - dp[0]
        cnts = [collections.Counter((nums[j] for j in xrange(i, len(nums), k))) for i in xrange(k)]
        return min(one_are_not_from_nums(nums, cnts), all_are_from_nums(nums, cnts))