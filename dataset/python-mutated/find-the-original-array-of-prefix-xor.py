class Solution(object):

    def findArray(self, pref):
        if False:
            return 10
        '\n        :type pref: List[int]\n        :rtype: List[int]\n        '
        for i in reversed(xrange(1, len(pref))):
            pref[i] ^= pref[i - 1]
        return pref