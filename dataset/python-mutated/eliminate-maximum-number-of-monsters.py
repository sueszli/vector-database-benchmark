class Solution(object):

    def eliminateMaximum(self, dist, speed):
        if False:
            i = 10
            return i + 15
        '\n        :type dist: List[int]\n        :type speed: List[int]\n        :rtype: int\n        '
        for i in xrange(len(dist)):
            dist[i] = (dist[i] - 1) // speed[i]
        dist.sort()
        result = 0
        for i in xrange(len(dist)):
            if result > dist[i]:
                break
            result += 1
        return result