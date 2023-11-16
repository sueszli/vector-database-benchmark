import collections

class Solution(object):

    def maxCandies(self, status, candies, keys, containedBoxes, initialBoxes):
        if False:
            i = 10
            return i + 15
        '\n        :type status: List[int]\n        :type candies: List[int]\n        :type keys: List[List[int]]\n        :type containedBoxes: List[List[int]]\n        :type initialBoxes: List[int]\n        :rtype: int\n        '
        result = 0
        q = collections.deque(initialBoxes)
        while q:
            changed = False
            for _ in xrange(len(q)):
                box = q.popleft()
                if not status[box]:
                    q.append(box)
                    continue
                changed = True
                result += candies[box]
                for contained_key in keys[box]:
                    status[contained_key] = 1
                for contained_box in containedBoxes[box]:
                    q.append(contained_box)
            if not changed:
                break
        return result