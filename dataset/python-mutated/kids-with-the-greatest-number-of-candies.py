class Solution(object):

    def kidsWithCandies(self, candies, extraCandies):
        if False:
            print('Hello World!')
        '\n        :type candies: List[int]\n        :type extraCandies: int\n        :rtype: List[bool]\n        '
        max_num = max(candies)
        return [x + extraCandies >= max_num for x in candies]