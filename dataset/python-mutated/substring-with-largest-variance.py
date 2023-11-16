import itertools

class Solution(object):

    def largestVariance(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '

        def modified_kadane(a, x, y):
            if False:
                i = 10
                return i + 15
            result = curr = 0
            lookup = [0] * 2
            remain = [a.count(x), a.count(y)]
            for c in a:
                if c not in (x, y):
                    continue
                lookup[c != x] = 1
                remain[c != x] -= 1
                curr += 1 if c == x else -1
                if curr < 0 and remain[0] and remain[1]:
                    curr = lookup[0] = lookup[1] = 0
                if lookup[0] and lookup[1]:
                    result = max(result, curr)
            return result
        alphabets = set(s)
        return max((modified_kadane(s, x, y) for (x, y) in itertools.permutations(alphabets, 2))) if len(alphabets) >= 2 else 0