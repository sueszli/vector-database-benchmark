class Solution(object):

    def splitLoopedString(self, strs):
        if False:
            i = 10
            return i + 15
        '\n        :type strs: List[str]\n        :rtype: str\n        '
        tmp = []
        for s in strs:
            tmp += max(s, s[::-1])
        s = ''.join(tmp)
        (result, st) = ('a', 0)
        for i in xrange(len(strs)):
            body = ''.join([s[st + len(strs[i]):], s[0:st]])
            for p in (strs[i], strs[i][::-1]):
                for j in xrange(len(strs[i])):
                    if p[j] >= result[0]:
                        result = max(result, ''.join([p[j:], body, p[:j]]))
            st += len(strs[i])
        return result