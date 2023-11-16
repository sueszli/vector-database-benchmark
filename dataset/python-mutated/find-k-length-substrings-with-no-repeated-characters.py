class Solution(object):

    def numKLenSubstrNoRepeats(self, S, K):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :type K: int\n        :rtype: int\n        '
        (result, i) = (0, 0)
        lookup = set()
        for j in xrange(len(S)):
            while S[j] in lookup:
                lookup.remove(S[i])
                i += 1
            lookup.add(S[j])
            result += j - i + 1 >= K
        return result