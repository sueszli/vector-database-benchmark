class Solution(object):

    def doesValidArrayExist(self, derived):
        if False:
            i = 10
            return i + 15
        '\n        :type derived: List[int]\n        :rtype: bool\n        '
        return reduce(lambda total, x: total ^ x, derived, 0) == 0