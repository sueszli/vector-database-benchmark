class Solution(object):

    def validSquare(self, p1, p2, p3, p4):
        if False:
            i = 10
            return i + 15
        '\n        :type p1: List[int]\n        :type p2: List[int]\n        :type p3: List[int]\n        :type p4: List[int]\n        :rtype: bool\n        '

        def dist(p1, p2):
            if False:
                print('Hello World!')
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        lookup = set([dist(p1, p2), dist(p1, p3), dist(p1, p4), dist(p2, p3), dist(p2, p4), dist(p3, p4)])
        return 0 not in lookup and len(lookup) == 2