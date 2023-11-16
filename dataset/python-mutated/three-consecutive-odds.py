class Solution(object):

    def threeConsecutiveOdds(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: bool\n        '
        count = 0
        for x in arr:
            count = count + 1 if x % 2 else 0
            if count == 3:
                return True
        return False