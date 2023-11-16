class Solution(object):

    def numOfBurgers(self, tomatoSlices, cheeseSlices):
        if False:
            print('Hello World!')
        '\n        :type tomatoSlices: int\n        :type cheeseSlices: int\n        :rtype: List[int]\n        '
        return [tomatoSlices // 2 - cheeseSlices, 2 * cheeseSlices - tomatoSlices // 2] if tomatoSlices % 2 == 0 and 2 * cheeseSlices <= tomatoSlices <= 4 * cheeseSlices else []