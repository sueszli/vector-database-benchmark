import collections

class Solution(object):

    def tallestBillboard(self, rods):
        if False:
            print('Hello World!')
        '\n        :type rods: List[int]\n        :rtype: int\n        '

        def dp(A):
            if False:
                for i in range(10):
                    print('nop')
            lookup = collections.defaultdict(int)
            lookup[0] = 0
            for x in A:
                for (d, y) in lookup.items():
                    lookup[d + x] = max(lookup[d + x], y)
                    lookup[abs(d - x)] = max(lookup[abs(d - x)], y + min(d, x))
            return lookup
        (left, right) = (dp(rods[:len(rods) // 2]), dp(rods[len(rods) // 2:]))
        return max((left[d] + right[d] + d for d in left if d in right))