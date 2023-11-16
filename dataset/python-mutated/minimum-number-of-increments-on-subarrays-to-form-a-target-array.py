class Solution(object):

    def minNumberOperations(self, target):
        if False:
            return 10
        '\n        :type target: List[int]\n        :rtype: int\n        '
        return target[0] + sum((max(target[i] - target[i - 1], 0) for i in xrange(1, len(target))))
import itertools

class Solution2(object):

    def minNumberOperations(self, target):
        if False:
            return 10
        '\n        :type target: List[int]\n        :rtype: int\n        '
        return sum((max(b - a, 0) for (b, a) in itertools.izip(target, [0] + target)))