import collections
import fractions

class Solution(object):

    def interchangeableRectangles(self, rectangles):
        if False:
            print('Hello World!')
        '\n        :type rectangles: List[List[int]]\n        :rtype: int\n        '
        count = collections.defaultdict(int)
        for (w, h) in rectangles:
            g = fractions.gcd(w, h)
            count[w // g, h // g] += 1
        return sum((c * (c - 1) // 2 for c in count.itervalues()))