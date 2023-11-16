class Solution(object):

    def asteroidsDestroyed(self, mass, asteroids):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type mass: int\n        :type asteroids: List[int]\n        :rtype: bool\n        '
        asteroids.sort()
        for x in asteroids:
            if x > mass:
                return False
            mass += min(x, asteroids[-1] - mass)
        return True