class Solution(object):

    def minSwaps(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        result = curr = 0
        for c in s:
            if c == ']':
                curr += 1
                result = max(result, curr)
            else:
                curr -= 1
        return (result + 1) // 2