class Solution(object):

    def captureForts(self, forts):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type forts: List[int]\n        :rtype: int\n        '
        result = left = 0
        for right in xrange(len(forts)):
            if not forts[right]:
                continue
            if forts[right] == -forts[left]:
                result = max(result, right - left - 1)
            left = right
        return result