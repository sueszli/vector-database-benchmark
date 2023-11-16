class Solution(object):

    def countBits(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: List[int]\n        '
        res = [0]
        for i in xrange(1, num + 1):
            res.append((i & 1) + res[i >> 1])
        return res

    def countBits2(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: List[int]\n        '
        s = [0]
        while len(s) <= num:
            s.extend(map(lambda x: x + 1, s))
        return s[:num + 1]