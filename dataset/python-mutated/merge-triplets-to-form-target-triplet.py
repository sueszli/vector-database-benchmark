class Solution(object):

    def mergeTriplets(self, triplets, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type triplets: List[List[int]]\n        :type target: List[int]\n        :rtype: bool\n        '
        result = [0] * 3
        for t in triplets:
            if all((t[i] <= target[i] for i in xrange(3))):
                result = [max(result[i], t[i]) for i in xrange(3)]
        return result == target