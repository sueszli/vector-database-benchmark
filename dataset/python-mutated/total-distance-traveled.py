class Solution(object):

    def distanceTraveled(self, mainTank, additionalTank):
        if False:
            while True:
                i = 10
        '\n        :type mainTank: int\n        :type additionalTank: int\n        :rtype: int\n        '
        (USE, REFILL, DIST) = (5, 1, 10)
        cnt = min((mainTank - REFILL) // (USE - REFILL), additionalTank)
        return (mainTank + cnt * REFILL) * DIST