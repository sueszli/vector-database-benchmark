class Solution(object):

    def maxTotalFruits(self, fruits, startPos, k):
        if False:
            i = 10
            return i + 15
        '\n        :type fruits: List[List[int]]\n        :type startPos: int\n        :type k: int\n        :rtype: int\n        '
        max_pos = max(startPos, fruits[-1][0])
        cnt = [0] * (1 + max_pos)
        for (p, a) in fruits:
            cnt[p] = a
        prefix = [0]
        for x in cnt:
            prefix.append(prefix[-1] + x)
        result = 0
        for left_dist in xrange(min(startPos, k) + 1):
            right_dist = max(k - 2 * left_dist, 0)
            (left, right) = (startPos - left_dist, min(startPos + right_dist, max_pos))
            result = max(result, prefix[right + 1] - prefix[left])
        for right_dist in xrange(min(max_pos - startPos, k) + 1):
            left_dist = max(k - 2 * right_dist, 0)
            (left, right) = (max(startPos - left_dist, 0), startPos + right_dist)
            result = max(result, prefix[right + 1] - prefix[left])
        return result