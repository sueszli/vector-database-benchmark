from itertools import izip

class Solution(object):

    def wordPattern(self, pattern, str):
        if False:
            print('Hello World!')
        '\n        :type pattern: str\n        :type str: str\n        :rtype: bool\n        '
        if len(pattern) != self.wordCount(str):
            return False
        (w2p, p2w) = ({}, {})
        for (p, w) in izip(pattern, self.wordGenerator(str)):
            if w not in w2p and p not in p2w:
                w2p[w] = p
                p2w[p] = w
            elif w not in w2p or w2p[w] != p:
                return False
        return True

    def wordCount(self, str):
        if False:
            print('Hello World!')
        cnt = 1 if str else 0
        for c in str:
            if c == ' ':
                cnt += 1
        return cnt

    def wordGenerator(self, str):
        if False:
            print('Hello World!')
        w = ''
        for c in str:
            if c == ' ':
                yield w
                w = ''
            else:
                w += c
        yield w

class Solution2(object):

    def wordPattern(self, pattern, str):
        if False:
            while True:
                i = 10
        '\n        :type pattern: str\n        :type str: str\n        :rtype: bool\n        '
        words = str.split()
        if len(pattern) != len(words):
            return False
        (w2p, p2w) = ({}, {})
        for (p, w) in izip(pattern, words):
            if w not in w2p and p not in p2w:
                w2p[w] = p
                p2w[p] = w
            elif w not in w2p or w2p[w] != p:
                return False
        return True