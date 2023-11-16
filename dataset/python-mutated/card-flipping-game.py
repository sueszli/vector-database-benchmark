import itertools

class Solution(object):

    def flipgame(self, fronts, backs):
        if False:
            print('Hello World!')
        '\n        :type fronts: List[int]\n        :type backs: List[int]\n        :rtype: int\n        '
        same = {n for (i, n) in enumerate(fronts) if n == backs[i]}
        result = float('inf')
        for n in itertools.chain(fronts, backs):
            if n not in same:
                result = min(result, n)
        return result if result < float('inf') else 0