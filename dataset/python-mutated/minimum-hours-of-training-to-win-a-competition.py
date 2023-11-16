import itertools

class Solution(object):

    def minNumberOfHours(self, initialEnergy, initialExperience, energy, experience):
        if False:
            return 10
        '\n        :type initialEnergy: int\n        :type initialExperience: int\n        :type energy: List[int]\n        :type experience: List[int]\n        :rtype: int\n        '
        result = 0
        for (hp, ex) in itertools.izip(energy, experience):
            inc1 = max(hp + 1 - initialEnergy, 0)
            inc2 = max(ex + 1 - initialExperience, 0)
            result += inc1 + inc2
            initialEnergy += inc1 - hp
            initialExperience += inc2 + ex
        return result