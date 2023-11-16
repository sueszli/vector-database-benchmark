class Solution(object):

    def greatestLetter(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: str\n        '
        lookup = set(s)
        result = ''
        for c in s:
            if c.isupper() and lower(c) in s:
                if c > result:
                    result = c
        return result
import itertools
import string

class Solution2(object):

    def greatestLetter(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: str\n        '
        lookup = set(s)
        return next((C for (c, C) in itertools.izip(reversed(string.ascii_lowercase), reversed(string.ascii_uppercase)) if c in lookup and C in lookup), '')