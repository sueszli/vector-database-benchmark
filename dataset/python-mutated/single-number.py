import operator
from functools import reduce

class Solution(object):
    """
    :type nums: List[int]
    :rtype: int
    """

    def singleNumber(self, A):
        if False:
            i = 10
            return i + 15
        return reduce(operator.xor, A)