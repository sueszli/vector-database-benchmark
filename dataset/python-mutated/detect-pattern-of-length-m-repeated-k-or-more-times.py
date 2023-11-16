class Solution(object):

    def containsPattern(self, arr, m, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :type m: int\n        :type k: int\n        :rtype: bool\n        '
        cnt = 0
        for i in xrange(len(arr) - m):
            if arr[i] != arr[i + m]:
                cnt = 0
                continue
            cnt += 1
            if cnt == (k - 1) * m:
                return True
        return False