class Solution(object):

    def wordPatternMatch(self, pattern, str):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type pattern: str\n        :type str: str\n        :rtype: bool\n        '
        (w2p, p2w) = ({}, {})
        return self.match(pattern, str, 0, 0, w2p, p2w)

    def match(self, pattern, str, i, j, w2p, p2w):
        if False:
            return 10
        is_match = False
        if i == len(pattern) and j == len(str):
            is_match = True
        elif i < len(pattern) and j < len(str):
            p = pattern[i]
            if p in p2w:
                w = p2w[p]
                if w == str[j:j + len(w)]:
                    is_match = self.match(pattern, str, i + 1, j + len(w), w2p, p2w)
            else:
                for k in xrange(j, len(str)):
                    w = str[j:k + 1]
                    if w not in w2p:
                        (w2p[w], p2w[p]) = (p, w)
                        is_match = self.match(pattern, str, i + 1, k + 1, w2p, p2w)
                        (w2p.pop(w), p2w.pop(p))
                    if is_match:
                        break
        return is_match