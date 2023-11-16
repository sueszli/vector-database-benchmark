class Solution(object):

    def freqAlphabets(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '

        def alpha(num):
            if False:
                print('Hello World!')
            return chr(ord('a') + int(num) - 1)
        i = 0
        result = []
        while i < len(s):
            if i + 2 < len(s) and s[i + 2] == '#':
                result.append(alpha(s[i:i + 2]))
                i += 3
            else:
                result.append(alpha(s[i]))
                i += 1
        return ''.join(result)

class Solution2(object):

    def freqAlphabets(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: str\n        '

        def alpha(num):
            if False:
                i = 10
                return i + 15
            return chr(ord('a') + int(num) - 1)
        i = len(s) - 1
        result = []
        while i >= 0:
            if s[i] == '#':
                result.append(alpha(s[i - 2:i]))
                i -= 3
            else:
                result.append(alpha(s[i]))
                i -= 1
        return ''.join(reversed(result))
import re

class Solution3(object):

    def freqAlphabets(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '

        def alpha(num):
            if False:
                i = 10
                return i + 15
            return chr(ord('a') + int(num) - 1)
        return ''.join((alpha(i[:2]) for i in re.findall('\\d\\d#|\\d', s)))