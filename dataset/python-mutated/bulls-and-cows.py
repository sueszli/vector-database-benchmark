import operator
from collections import defaultdict, Counter
from itertools import izip, imap

class Solution(object):

    def getHint(self, secret, guess):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type secret: str\n        :type guess: str\n        :rtype: str\n        '
        (A, B) = (0, 0)
        lookup = defaultdict(int)
        for (s, g) in izip(secret, guess):
            if s == g:
                A += 1
            else:
                B += int(lookup[s] < 0) + int(lookup[g] > 0)
                lookup[s] += 1
                lookup[g] -= 1
        return '%dA%dB' % (A, B)

class Solution2(object):

    def getHint(self, secret, guess):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type secret: str\n        :type guess: str\n        :rtype: str\n        '
        A = sum(imap(operator.eq, secret, guess))
        B = sum((Counter(secret) & Counter(guess)).values()) - A
        return '%dA%dB' % (A, B)