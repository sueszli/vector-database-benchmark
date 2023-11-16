class Solution(object):

    def platesBetweenCandles(self, s, queries):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        (left, prefix) = ([0] * len(s), {})
        (curr, cnt) = (-1, 0)
        for i in xrange(len(s)):
            if s[i] == '|':
                curr = i
                cnt += 1
                prefix[i] = cnt
            left[i] = curr
        right = [0] * len(s)
        curr = len(s)
        for i in reversed(xrange(len(s))):
            if s[i] == '|':
                curr = i
            right[i] = curr
        return [max(left[r] - right[l] + 1 - (prefix[left[r]] - prefix[right[l]] + 1), 0) for (l, r) in queries]