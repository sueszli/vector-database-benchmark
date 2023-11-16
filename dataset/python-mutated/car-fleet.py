class Solution(object):

    def carFleet(self, target, position, speed):
        if False:
            print('Hello World!')
        '\n        :type target: int\n        :type position: List[int]\n        :type speed: List[int]\n        :rtype: int\n        '
        times = [float(target - p) / s for (p, s) in sorted(zip(position, speed))]
        (result, curr) = (0, 0)
        for t in reversed(times):
            if t > curr:
                result += 1
                curr = t
        return result