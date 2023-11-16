class Solution(object):

    def minimumPartition(self, s, k):
        if False:
            return 10
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        result = 1
        curr = 0
        for c in s:
            if int(c) > k:
                return -1
            if curr * 10 + int(c) > k:
                result += 1
                curr = 0
            curr = curr * 10 + int(c)
        return result