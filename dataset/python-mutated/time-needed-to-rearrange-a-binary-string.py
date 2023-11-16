class Solution(object):

    def secondsToRemoveOccurrences(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = cnt = 0
        for c in s:
            if c == '0':
                cnt += 1
                continue
            if cnt:
                result = max(result + 1, cnt)
        return result