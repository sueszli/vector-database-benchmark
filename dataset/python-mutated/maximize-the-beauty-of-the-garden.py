class Solution(object):

    def maximumBeauty(self, flowers):
        if False:
            i = 10
            return i + 15
        '\n        :type flowers: List[int]\n        :rtype: int\n        '
        lookup = {}
        prefix = [0]
        result = float('-inf')
        for (i, f) in enumerate(flowers):
            prefix.append(prefix[-1] + f if f > 0 else prefix[-1])
            if not f in lookup:
                lookup[f] = i
                continue
            result = max(result, 2 * f + prefix[i + 1] - prefix[lookup[f]] if f < 0 else prefix[i + 1] - prefix[lookup[f]])
        return result