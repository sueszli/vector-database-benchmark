class Solution(object):

    def longestCommonPrefix(self, strs):
        if False:
            while True:
                i = 10
        '\n        :type strs: List[str]\n        :rtype: str\n        '
        if not strs:
            return ''
        for i in xrange(len(strs[0])):
            for string in strs[1:]:
                if i >= len(string) or string[i] != strs[0][i]:
                    return strs[0][:i]
        return strs[0]

class Solution2(object):

    def longestCommonPrefix(self, strs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type strs: List[str]\n        :rtype: str\n        '
        prefix = ''
        for chars in zip(*strs):
            if all((c == chars[0] for c in chars)):
                prefix += chars[0]
            else:
                return prefix
        return prefix