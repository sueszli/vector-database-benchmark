import collections

class Solution(object):

    def longestSubsequenceRepeatedK(self, s, k):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type k: int\n        :rtype: str\n        '

        def check(s, k, curr):
            if False:
                print('Hello World!')
            if not curr:
                return True
            i = 0
            for c in s:
                if c != curr[i]:
                    continue
                i += 1
                if i != len(curr):
                    continue
                i = 0
                k -= 1
                if not k:
                    return True
            return False

        def backtracking(s, k, curr, cnts, result):
            if False:
                i = 10
                return i + 15
            if not check(s, k, curr):
                return
            if len(curr) > len(result):
                result[:] = curr
            for c in reversed(string.ascii_lowercase):
                if cnts[c] < k:
                    continue
                cnts[c] -= k
                curr.append(c)
                backtracking(s, k, curr, cnts, result)
                curr.pop()
                cnts[c] += k
        cnts = collections.Counter(s)
        new_s = []
        for c in s:
            if cnts[c] < k:
                continue
            new_s.append(c)
        result = []
        backtracking(new_s, k, [], cnts, result)
        return ''.join(result)