class Solution(object):

    def escapeGhosts(self, ghosts, target):
        if False:
            return 10
        '\n        :type ghosts: List[List[int]]\n        :type target: List[int]\n        :rtype: bool\n        '
        total = abs(target[0]) + abs(target[1])
        return all((total < abs(target[0] - i) + abs(target[1] - j) for (i, j) in ghosts))