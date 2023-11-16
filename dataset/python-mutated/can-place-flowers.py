class Solution(object):

    def canPlaceFlowers(self, flowerbed, n):
        if False:
            while True:
                i = 10
        '\n        :type flowerbed: List[int]\n        :type n: int\n        :rtype: bool\n        '
        for i in xrange(len(flowerbed)):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0):
                flowerbed[i] = 1
                n -= 1
            if n <= 0:
                return True
        return False