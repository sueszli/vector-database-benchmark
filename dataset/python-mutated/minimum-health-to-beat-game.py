class Solution(object):

    def minimumHealth(self, damage, armor):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type damage: List[int]\n        :type armor: int\n        :rtype: int\n        '
        return sum(damage) - min(max(damage), armor) + 1