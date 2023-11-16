class Solution(object):

    def findReplaceString(self, S, indexes, sources, targets):
        if False:
            return 10
        '\n        :type S: str\n        :type indexes: List[int]\n        :type sources: List[str]\n        :type targets: List[str]\n        :rtype: str\n        '
        bucket = [None] * len(S)
        for i in xrange(len(indexes)):
            if all((indexes[i] + k < len(S) and S[indexes[i] + k] == sources[i][k] for k in xrange(len(sources[i])))):
                bucket[indexes[i]] = (len(sources[i]), list(targets[i]))
        result = []
        i = 0
        while i < len(S):
            if bucket[i]:
                result.extend(bucket[i][1])
                i += bucket[i][0]
            else:
                result.append(S[i])
                i += 1
        return ''.join(result)

class Solution2(object):

    def findReplaceString(self, S, indexes, sources, targets):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :type indexes: List[int]\n        :type sources: List[str]\n        :type targets: List[str]\n        :rtype: str\n        '
        for (i, s, t) in sorted(zip(indexes, sources, targets), reverse=True):
            if S[i:i + len(s)] == s:
                S = S[:i] + t + S[i + len(s):]
        return S