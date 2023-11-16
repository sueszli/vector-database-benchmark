import collections

class Solution(object):

    def findRepeatedDnaSequences(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: List[str]\n        '
        (dict, rolling_hash, res) = ({}, 0, [])
        for i in xrange(len(s)):
            rolling_hash = rolling_hash << 3 & 1073741823 | ord(s[i]) & 7
            if rolling_hash not in dict:
                dict[rolling_hash] = True
            elif dict[rolling_hash]:
                res.append(s[i - 9:i + 1])
                dict[rolling_hash] = False
        return res

    def findRepeatedDnaSequences2(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: List[str]\n        '
        (l, r) = ([], [])
        if len(s) < 10:
            return []
        for i in range(len(s) - 9):
            l.extend([s[i:i + 10]])
        return [k for (k, v) in collections.Counter(l).items() if v > 1]