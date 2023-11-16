class Solution(object):

    def carPooling(self, trips, capacity):
        if False:
            return 10
        '\n        :type trips: List[List[int]]\n        :type capacity: int\n        :rtype: bool\n        '
        line = [x for (num, start, end) in trips for x in [[start, num], [end, -num]]]
        line.sort()
        for (_, num) in line:
            capacity -= num
            if capacity < 0:
                return False
        return True