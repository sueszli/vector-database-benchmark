import collections

class Solution(object):

    def meetRequirement(self, n, lights, requirement):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type lights: List[List[int]]\n        :type requirement: List[int]\n        :rtype: int\n        '
        cnt = collections.defaultdict(int)
        for (p, r) in lights:
            cnt[max(p - r, 0)] += 1
            cnt[min(p + r, n - 1) + 1] -= 1
        result = curr = 0
        for (i, r) in enumerate(requirement):
            curr += cnt.get(i, 0)
            if curr >= r:
                result += 1
        return result