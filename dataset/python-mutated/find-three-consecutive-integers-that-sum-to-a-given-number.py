class Solution(object):

    def sumOfThree(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: List[int]\n        '
        return [num // 3 - 1, num // 3, num // 3 + 1] if num % 3 == 0 else []