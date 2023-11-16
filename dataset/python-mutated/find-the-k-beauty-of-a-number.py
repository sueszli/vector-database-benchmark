class Solution(object):

    def divisorSubstrings(self, num, k):
        if False:
            print('Hello World!')
        '\n        :type num: int\n        :type k: int\n        :rtype: int\n        '
        result = curr = 0
        s = map(int, str(num))
        base = 10 ** (k - 1)
        for (i, x) in enumerate(s):
            if i - k >= 0:
                curr -= s[i - k] * base
            curr = curr * 10 + x
            if i + 1 >= k:
                result += int(curr and num % curr == 0)
        return result