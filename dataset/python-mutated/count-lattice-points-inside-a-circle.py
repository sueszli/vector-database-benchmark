class Solution(object):

    def countLatticePoints(self, circles):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type circles: List[List[int]]\n        :rtype: int\n        '
        lookup = set()
        for (x, y, r) in circles:
            for i in xrange(-r, r + 1):
                for j in xrange(-r, r + 1):
                    if i ** 2 + j ** 2 <= r ** 2:
                        lookup.add((x + i, y + j))
        return len(lookup)

class Solution2(object):

    def countLatticePoints(self, circles):
        if False:
            return 10
        '\n        :type circles: List[List[int]]\n        :rtype: int\n        '
        max_x = max((x + r for (x, _, r) in circles))
        max_y = max((y + r for (_, y, r) in circles))
        result = 0
        for i in xrange(max_x + 1):
            for j in xrange(max_y + 1):
                if any(((i - x) ** 2 + (j - y) ** 2 <= r ** 2 for (x, y, r) in circles)):
                    result += 1
        return result