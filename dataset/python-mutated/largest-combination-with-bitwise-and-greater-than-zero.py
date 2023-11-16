class Solution(object):

    def largestCombination(self, candidates):
        if False:
            i = 10
            return i + 15
        '\n        :type candidates: List[int]\n        :rtype: int\n        '
        cnt = []
        (base, mx) = (1, max(candidates))
        while base <= mx:
            cnt.append(sum((x & base > 0 for x in candidates)))
            base <<= 1
        return max(cnt)