class Solution(object):

    def isRectangleOverlap(self, rec1, rec2):
        if False:
            i = 10
            return i + 15
        '\n        :type rec1: List[int]\n        :type rec2: List[int]\n        :rtype: bool\n        '

        def intersect(p_left, p_right, q_left, q_right):
            if False:
                for i in range(10):
                    print('nop')
            return max(p_left, q_left) < min(p_right, q_right)
        return intersect(rec1[0], rec1[2], rec2[0], rec2[2]) and intersect(rec1[1], rec1[3], rec2[1], rec2[3])