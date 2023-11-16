import itertools

class Solution(object):

    def maximumCostSubstring(self, s, chars, vals):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type chars: str\n        :type vals: List[int]\n        :rtype: int\n        '

        def kadane(s):
            if False:
                i = 10
                return i + 15
            result = curr = 0
            for c in s:
                curr = max(curr + (lookup[c] if c in lookup else ord(c) - ord('a') + 1), 0)
                result = max(result, curr)
            return result
        lookup = {}
        for (c, v) in itertools.izip(chars, vals):
            lookup[c] = v
        return kadane(s)