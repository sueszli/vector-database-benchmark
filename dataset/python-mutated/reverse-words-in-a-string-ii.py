class Solution(object):

    def reverseWords(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: a list of 1 length strings (List[str])\n        :rtype: nothing\n        '

        def reverse(s, begin, end):
            if False:
                return 10
            for i in xrange((end - begin) / 2):
                (s[begin + i], s[end - 1 - i]) = (s[end - 1 - i], s[begin + i])
        reverse(s, 0, len(s))
        i = 0
        for j in xrange(len(s) + 1):
            if j == len(s) or s[j] == ' ':
                reverse(s, i, j)
                i = j + 1