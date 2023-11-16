class Solution(object):

    def minimumJumps(self, forbidden, a, b, x):
        if False:
            return 10
        '\n        :type forbidden: List[int]\n        :type a: int\n        :type b: int\n        :type x: int\n        :rtype: int\n        '
        max_f = max(forbidden)
        max_val = x + b if a >= b else max(x, max_f) + a + (b + a)
        lookup = set()
        for pos in forbidden:
            lookup.add((pos, True))
            lookup.add((pos, False))
        result = 0
        q = [(0, True)]
        lookup.add((0, True))
        while q:
            new_q = []
            for (pos, can_back) in q:
                if pos == x:
                    return result
                if pos + a <= max_val and (pos + a, True) not in lookup:
                    lookup.add((pos + a, True))
                    new_q.append((pos + a, True))
                if not can_back:
                    continue
                if pos - b >= 0 and (pos - b, False) not in lookup:
                    lookup.add((pos - b, False))
                    new_q.append((pos - b, False))
            q = new_q
            result += 1
        return -1