class Solution(object):

    def isTransformable(self, s, t):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        idxs = [[] for _ in xrange(10)]
        for i in reversed(xrange(len(s))):
            idxs[int(s[i])].append(i)
        for c in t:
            d = int(c)
            if not idxs[d]:
                return False
            for k in xrange(d):
                if idxs[k] and idxs[k][-1] < idxs[d][-1]:
                    return False
            idxs[d].pop()
        return True