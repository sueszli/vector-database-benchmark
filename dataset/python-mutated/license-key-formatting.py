class Solution(object):

    def licenseKeyFormatting(self, S, K):
        if False:
            return 10
        '\n        :type S: str\n        :type K: int\n        :rtype: str\n        '
        result = []
        for i in reversed(xrange(len(S))):
            if S[i] == '-':
                continue
            if len(result) % (K + 1) == K:
                result += '-'
            result += S[i].upper()
        return ''.join(reversed(result))