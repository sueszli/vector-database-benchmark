class Solution(object):

    def maximumGroups(self, grades):
        if False:
            return 10
        '\n        :type grades: List[int]\n        :rtype: int\n        '
        return int(((1 + 8 * len(grades)) ** 0.5 - 1) / 2.0)